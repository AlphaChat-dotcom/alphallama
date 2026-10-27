#pragma once
// Tiered, packed backing store for the native `--context unified` index (server-context.cpp).
//
// The resident dict index holds the full record text (unified_index::entity), twice, plus a
// per-record std::unordered_map extract (rec_kv/groups/rec_deps/rec_type/rec_section) with large
// container overhead, which does not scale to large corpora.
//
// This splits it into a paged disk tier + a packed resident tier (mirrors tiered_index.py):
//   - records.bin (mmap, ~0 resident): each record's block written once; spans in rec_off int64[n+1].
//   - name arena: ids are sorted-name rank; name_blob + name_off int32[n+1]; find_id = binary search
//     (no resident name->id hash).
//   - interned columns (uint16 codes + small vocabs), kv CSR, groups inverted index (int32 ids),
//     edges CSR.
// Same query output as the dict path; group id-lists are emitted in ascending-id (= sorted-name)
// order, so it is deterministic (the dict path uses unordered_map order; scores are identical).
//
// Built once from the transient build-time maps in unified_index::finalize(), which are then freed.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdio>

#ifndef _WIN32
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

struct tiered_store {
    // Owns an mmap + fd -> non-copyable (a copy would double-munmap / dangle `mm`). Never copied in
    // practice (a long-lived member of the server's unified_index).
    tiered_store() = default;
    tiered_store(const tiered_store&) = delete;
    tiered_store& operator=(const tiered_store&) = delete;

    int n = 0;
    bool ready = false;

    // ---- name arena (ids = sorted-name rank) ----
    std::string          name_blob;
    std::vector<int32_t> name_off;      // n+1

    // ---- records.bin (mmap) ----
    std::vector<int64_t> rec_off;       // n+1 ; block i = mm[rec_off[i] .. rec_off[i+1])
    std::string          records_path;
    const unsigned char* mm = nullptr;
    size_t               mm_len = 0;
#ifndef _WIN32
    int                  fd = -1;
#endif
    std::string          fallback_text; // used when mmap unavailable (e.g. _WIN32 dev builds)

    // ---- interned single-value columns ----
    std::vector<uint16_t>    type_code;   // n ; 0xFFFF = none
    std::vector<uint16_t>    sect_code;   // n ; 0xFFFF = none
    std::vector<std::string> type_vocab;  // code -> string
    std::vector<std::string> sect_vocab;  // code -> string

    // ---- groups inverted index (flat columnar): one row per distinct (key,value) with an id-list ----
    // Special synthetic keys "_type"/"_section" are stored here too (as their literal key strings), so
    // the query branches that iterate/skip keys prefixed with '_' behave exactly as before.
    std::vector<std::string> grp_key;      // G ; key string (e.g. "birth_year", "_section", "_type")
    std::vector<std::string> grp_val;      // G ; value string
    std::vector<int32_t>     grp_ids_ptr;  // G+1 ; into grp_ids
    std::vector<int32_t>     grp_ids;      // sum ; ascending ids

    // ---- edges CSR ----
    std::vector<int32_t> edge_ptr;         // n+1
    std::vector<int32_t> edge_tgt;         // E ; ascending? (kept in resolution order to match dict)

    // ---- val_count (sorted arena + counts; binary-search) ----
    std::string          vc_blob;
    std::vector<int32_t> vc_off;           // m+1
    std::vector<int32_t> vc_cnt;           // m
    int                  vc_n = 0;

    // build
    // Consumes the transient dict maps from unified_index. `records_dir` is where records.bin is
    // written (the corpus/session dir). Returns false on I/O failure (caller keeps the dict path).
    bool build(const std::string & records_path_in,   // full path for records.bin ("" -> in-memory)
               const std::unordered_map<std::string, std::string> & entity,
               const std::unordered_map<std::string, std::vector<std::pair<std::string,std::string>>> & rec_kv,
               const std::unordered_map<std::string, std::string> & rec_type,
               const std::unordered_map<std::string, std::string> & rec_section,
               const std::unordered_map<std::string, std::vector<std::string>> & rec_deps,
               const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> & groups,
               const std::unordered_map<std::string, int> & val_count)
    {
        // 1) id = sorted-name rank over the union of record ids (entity keys are the canonical set).
        std::vector<std::string> names;
        names.reserve(entity.size());
        for (const auto & e : entity) names.push_back(e.first);
        std::sort(names.begin(), names.end());
        n = (int)names.size();
        std::unordered_map<std::string,int> id_of;   // build-time only, discarded
        id_of.reserve(n * 2);
        for (int i = 0; i < n; i++) id_of[names[i]] = i;

        // 2) name arena
        name_off.assign(n + 1, 0);
        for (int i = 0; i < n; i++) { name_blob += names[i]; name_off[i+1] = (int32_t)name_blob.size(); }

        // 3) records.bin - write each block once, in id order; record spans. If the file can't be
        //    written (read-only dir / no path), fall back to an in-memory arena so the query path is
        //    still single (always via this tier); only the mmap RAM-saving is lost in that case.
        records_path = records_path_in;
        rec_off.assign(n + 1, 0);
        FILE * f = records_path.empty() ? nullptr : fopen(records_path.c_str(), "wb");
        bool to_file = (f != nullptr);
        {
            int64_t off = 0;
            for (int i = 0; i < n; i++) {
                const std::string & blk = entity.at(names[i]);
                if (to_file) {
                    if (!blk.empty() && fwrite(blk.data(), 1, blk.size(), f) != blk.size()) {
                        fclose(f); f = nullptr; to_file = false; fallback_text.clear();
                        // restart into memory (rare): re-append everything from record 0
                        for (int j = 0; j <= i; j++) fallback_text += entity.at(names[j]);
                    }
                } else {
                    fallback_text += blk;
                }
                off += (int64_t)blk.size();
                rec_off[i+1] = off;
            }
            if (f) fclose(f);
        }

        // 4) interned single-value columns (type, section)
        type_code.assign(n, 0xFFFF);
        sect_code.assign(n, 0xFFFF);
        { std::unordered_map<std::string,uint16_t> voc;
          for (const auto & kv : rec_type) { auto it=id_of.find(kv.first); if(it==id_of.end()||kv.second.empty())continue;
              auto v=voc.find(kv.second); uint16_t c; if(v==voc.end()){ c=(uint16_t)type_vocab.size(); voc[kv.second]=c; type_vocab.push_back(kv.second);} else c=v->second;
              type_code[it->second]=c; } }
        { std::unordered_map<std::string,uint16_t> voc;
          for (const auto & kv : rec_section) { auto it=id_of.find(kv.first); if(it==id_of.end()||kv.second.empty())continue;
              auto v=voc.find(kv.second); uint16_t c; if(v==voc.end()){ c=(uint16_t)sect_vocab.size(); voc[kv.second]=c; sect_vocab.push_back(kv.second);} else c=v->second;
              sect_code[it->second]=c; } }

        // 5) groups inverted index -> flat rows, ids remapped to int + sorted ascending.
        grp_ids_ptr.push_back(0);
        for (const auto & g : groups) {
            for (const auto & vv : g.second) {
                std::vector<int32_t> ids;
                ids.reserve(vv.second.size());
                for (const auto & sid : vv.second) { auto it=id_of.find(sid); if(it!=id_of.end()) ids.push_back(it->second); }
                if (ids.empty()) continue;
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                grp_key.push_back(g.first);
                grp_val.push_back(vv.first);
                for (int32_t id : ids) grp_ids.push_back(id);
                grp_ids_ptr.push_back((int32_t)grp_ids.size());
            }
        }

        // 6) edges CSR (from rec_deps; keep per-record resolution order to match the dict's dep order)
        edge_ptr.assign(n + 1, 0);
        {
            std::vector<std::vector<int32_t>> per(n);
            for (const auto & d : rec_deps) {
                auto it = id_of.find(d.first); if (it==id_of.end()) continue;
                auto & dst = per[it->second];
                for (const auto & t : d.second) { auto jt=id_of.find(t); if(jt!=id_of.end()) dst.push_back(jt->second); }
            }
            for (int i = 0; i < n; i++) { for (int32_t t : per[i]) edge_tgt.push_back(t); edge_ptr[i+1]=(int32_t)edge_tgt.size(); }
        }

        // 7) val_count -> sorted arena + counts
        {
            std::vector<std::pair<std::string,int>> vc(val_count.begin(), val_count.end());
            std::sort(vc.begin(), vc.end(), [](const auto&a, const auto&b){ return a.first<b.first; });
            vc_n = (int)vc.size();
            vc_off.assign(vc_n + 1, 0);
            for (int i = 0; i < vc_n; i++) { vc_blob += vc[i].first; vc_off[i+1]=(int32_t)vc_blob.size(); vc_cnt.push_back(vc[i].second); }
        }

        // 8) point `mm` at the text: mmap the file, or the in-memory fallback arena.
        if (to_file) {
            if (!map_records()) {   // mmap failed -> slurp the just-written file into memory
                FILE * rf = fopen(records_path.c_str(), "rb");
                if (rf) { fseek(rf,0,SEEK_END); long sz=ftell(rf); fseek(rf,0,SEEK_SET);
                          fallback_text.resize(sz>0?sz:0); if(sz>0) { size_t rd=fread(&fallback_text[0],1,sz,rf); (void)rd; } fclose(rf); }
                mm = (const unsigned char*)fallback_text.data(); mm_len = fallback_text.size();
            }
        } else {
            mm = (const unsigned char*)fallback_text.data();
            mm_len = fallback_text.size();
        }
        ready = true;
        return true;
    }

    bool map_records() {
#ifndef _WIN32
        fd = open(records_path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st) != 0) { close(fd); fd=-1; return false; }
        mm_len = (size_t)st.st_size;
        if (mm_len == 0) { mm = (const unsigned char*)""; return true; }
        void * p = mmap(nullptr, mm_len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { close(fd); fd=-1; return false; }
        mm = (const unsigned char*)p;
        // Unlink now: the mapping stays valid after the directory entry is gone (POSIX), so there is
        // no on-disk residue and no chance a re-ingest lists records.bin as a corpus file.
        if (!records_path.empty()) unlink(records_path.c_str());
        return true;
#else
        // Dev/_WIN32 fallback: slurp into resident memory (servers are Linux -> mmap path).
        FILE * f = fopen(records_path.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        fallback_text.resize(sz > 0 ? sz : 0);
        if (sz > 0 && fread(&fallback_text[0], 1, sz, f) != (size_t)sz) { fclose(f); return false; }
        fclose(f);
        mm = (const unsigned char*)fallback_text.data();
        mm_len = fallback_text.size();
        return true;
#endif
    }

    ~tiered_store() {
#ifndef _WIN32
        if (mm && mm_len && fd >= 0) munmap((void*)mm, mm_len);
        if (fd >= 0) close(fd);
#endif
    }

    // accessors
    std::string_view name_of(int id) const {
        return std::string_view(name_blob.data() + name_off[id], (size_t)(name_off[id+1] - name_off[id]));
    }
    std::string name_str(int id) const { auto v = name_of(id); return std::string(v.data(), v.size()); }

    // binary search the sorted name arena; -1 if absent. Bytewise compare (matches std::string <).
    int find_id(const std::string & name) const {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            std::string_view cur = name_of(mid);
            int c = cur.compare(std::string_view(name));
            if (c == 0) return mid;
            if (c < 0) lo = mid + 1; else hi = mid;
        }
        return -1;
    }
    bool has(const std::string & name) const { return find_id(name) >= 0; }

    std::string block_of(int id) const {
        int64_t a = rec_off[id], b = rec_off[id+1];
        if (b <= a) return std::string();
        return std::string((const char*)mm + a, (size_t)(b - a));
    }

    // group-row accessors (flat inverted index)
    int gn() const { return (int)grp_key.size(); }
    int gids_n(int g) const { return grp_ids_ptr[g+1] - grp_ids_ptr[g]; }

    // Resident bytes = the columnar/arena tier only. Record text is on mmap'd disk (records.bin) and
    // does not count; it counts only if the in-memory fallback kicked in.
    size_t resident_bytes() const {
        size_t b = name_blob.size() + name_off.size()*4 + rec_off.size()*8
                 + type_code.size()*2 + sect_code.size()*2
                 + grp_ids_ptr.size()*4 + grp_ids.size()*4
                 + edge_ptr.size()*4 + edge_tgt.size()*4
                 + vc_blob.size() + vc_off.size()*4 + vc_cnt.size()*4
                 + fallback_text.size();   // 0 when records.bin was mmap'd from disk
        auto vecstr = [](const std::vector<std::string>& v){ size_t s=0; for(auto&x:v) s+=x.size()+sizeof(std::string); return s; };
        b += vecstr(grp_key) + vecstr(grp_val) + vecstr(type_vocab) + vecstr(sect_vocab);
        return b;
    }

    std::string type_of(int id) const { uint16_t c = type_code[id]; return c==0xFFFF ? std::string() : type_vocab[c]; }
    std::string sect_of(int id) const { uint16_t c = sect_code[id]; return c==0xFFFF ? std::string() : sect_vocab[c]; }

    // edges
    std::pair<const int32_t*, const int32_t*> deps_of(int id) const {
        return { edge_tgt.data() + edge_ptr[id], edge_tgt.data() + edge_ptr[id+1] };
    }
    bool has_deps(int id) const { return edge_ptr[id+1] > edge_ptr[id]; }

    // val_count lookup
    int val_count_of(const std::string & v) const {
        int lo = 0, hi = vc_n;
        while (lo < hi) { int mid=(lo+hi)>>1;
            std::string_view cur(vc_blob.data()+vc_off[mid], (size_t)(vc_off[mid+1]-vc_off[mid]));
            int c = cur.compare(std::string_view(v));
            if (c==0) return vc_cnt[mid];
            if (c<0) lo=mid+1; else hi=mid;
        }
        return 0;
    }
};
