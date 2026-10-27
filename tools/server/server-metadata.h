#pragma once

// Metadata Sub-Index: schema-free structured record indexing.
// Auto-detects 5 field types at ingest: person, date, location, numeric, category.
// Query-time: intersect posting lists -> sub-record precision (1K not 16K).
// Works on any domain: insurance, medical, legal, financial, code, logs, etc.

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <regex>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#ifndef _WIN32
#include <unistd.h>   // unlink
#endif

// A single record within a chunk
struct record_entry {
    int chunk_id;
    int byte_offset;
    int byte_length;
};

// Sorted index for range queries (dates, numeric values)
struct range_index {
    struct entry {
        double value;
        record_entry record;
    };
    std::vector<entry> entries;  // kept sorted by value
    bool sorted = false;

    void add(double value, const record_entry & rec) {
        entries.push_back({value, rec});
        sorted = false;
    }

    void ensure_sorted() {
        if (!sorted) {
            std::sort(entries.begin(), entries.end(),
                [](const entry & a, const entry & b) { return a.value < b.value; });
            sorted = true;
        }
    }

    std::vector<record_entry> range(double min_val, double max_val) {
        ensure_sorted();
        std::vector<record_entry> result;
        auto lo = std::lower_bound(entries.begin(), entries.end(), min_val,
            [](const entry & e, double v) { return e.value < v; });
        for (auto it = lo; it != entries.end() && it->value <= max_val; ++it) {
            result.push_back(it->record);
        }
        return result;
    }
};

// Inverted index for exact-match fields (person, location, category).
// At build time postings live in a map. After ingest, finalize() collapses it into a sorted key
// arena + CSR of record_entry (binary-search lookup) and frees the map, which would otherwise
// dominate resident memory. lookup() falls back to the map if not finalized (incremental API path).
struct inverted_index {
    std::unordered_map<std::string, std::vector<record_entry>> posting;   // build-time (incremental / selfcheck)
    // string-keyed packed CSR (incremental finalize path):
    std::string               key_blob;     // sorted keys concatenated
    std::vector<int32_t>      key_off;      // nkeys+1 offsets into key_blob
    std::vector<int32_t>      rec_ptr;      // nkeys+1 CSR row pointers
    std::vector<record_entry> rec_data;     // CSR payload
    bool                      finalized = false;

    // Streaming (batch) mode: add() writes hashed (key_hash, record_entry) records to a temp file
    // instead of a map, and finalize radix-bucket sorts them into a hash-keyed CSR, so build peak is
    // the CSR output plus one bucket. lookup() hashes the key and binary-searches (64-bit hash, so
    // collisions are negligible). selfcheck keeps the map too and verifies every key.
    bool                      streaming = false;
    bool                      selfcheck = false;
    FILE *                    sfp = nullptr;   // stream write handle (open during ingest)
    std::string               spath;
    std::vector<uint64_t>     hkeys;           // sorted distinct key hashes
    std::vector<int32_t>      hptr;            // nkeys+1 CSR
    std::vector<record_entry> hrec;            // CSR payload
    bool                      hfinalized = false;

    static uint64_t hashkey(const std::string & k) {   // FNV-1a 64
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char c : k) { h ^= c; h *= 1099511628211ULL; }
        return h;
    }

    void open_stream(const std::string & path, bool self) {
        spath = path; sfp = fopen(path.c_str(), "wb");
        streaming = (sfp != nullptr); selfcheck = self;
    }

    void add(const std::string & key, const record_entry & rec) {
        if (streaming && sfp) {
            uint64_t h = hashkey(key);
            fwrite(&h, 8, 1, sfp);
            fwrite(&rec, sizeof(record_entry), 1, sfp);   // 12 B
        }
        if (!streaming || selfcheck) posting[key].push_back(rec);
    }

    void finalize() {
        if (streaming) { finalize_stream(); return; }
        if (finalized || posting.empty()) return;
        std::vector<const std::pair<const std::string, std::vector<record_entry>>*> items;
        items.reserve(posting.size());
        for (auto & kv : posting) items.push_back(&kv);
        std::sort(items.begin(), items.end(), [](auto a, auto b){ return a->first < b->first; });
        key_off.push_back(0); rec_ptr.push_back(0);
        for (auto * kv : items) {
            key_blob += kv->first;
            key_off.push_back((int32_t)key_blob.size());
            for (auto & r : kv->second) rec_data.push_back(r);
            rec_ptr.push_back((int32_t)rec_data.size());
        }
        std::unordered_map<std::string, std::vector<record_entry>>().swap(posting);   // free
        finalized = true;
    }

    // Radix-bucket external sort of the stream file -> hash-keyed CSR. Bounded memory (one bucket).
    void finalize_stream() {
        if (hfinalized) return;
        if (sfp) { fclose(sfp); sfp = nullptr; }
        struct rec20 { uint64_t h; record_entry r; };
        const int KBITS = 6, K = 1 << KBITS;   // 64 range-buckets by top bits, so buckets are hash-ordered
        std::vector<FILE*> bf(K, nullptr);
        for (int b = 0; b < K; b++) bf[b] = fopen((spath + ".b" + std::to_string(b)).c_str(), "wb");
        if (FILE * in = fopen(spath.c_str(), "rb")) {
            uint64_t h; record_entry r;
            while (fread(&h, 8, 1, in) == 1 && fread(&r, sizeof(record_entry), 1, in) == 1) {
                int b = (int)(h >> (64 - KBITS));
                if (bf[b]) { fwrite(&h, 8, 1, bf[b]); fwrite(&r, sizeof(record_entry), 1, bf[b]); }
            }
            fclose(in);
        }
        for (int b = 0; b < K; b++) if (bf[b]) fclose(bf[b]);
#ifndef _WIN32
        unlink(spath.c_str());
#endif
        hptr.push_back(0);
        for (int b = 0; b < K; b++) {
            std::string bp = spath + ".b" + std::to_string(b);
            FILE * f = fopen(bp.c_str(), "rb");
            if (!f) continue;
            std::vector<rec20> recs;
            { uint64_t h; record_entry r;
              while (fread(&h, 8, 1, f) == 1 && fread(&r, sizeof(record_entry), 1, f) == 1) recs.push_back({h, r}); }
            fclose(f);
#ifndef _WIN32
            unlink(bp.c_str());
#endif
            std::sort(recs.begin(), recs.end(), [](const rec20 & a, const rec20 & b){ return a.h < b.h; });
            for (size_t i = 0; i < recs.size(); ) {
                size_t j = i; while (j < recs.size() && recs[j].h == recs[i].h) j++;
                hkeys.push_back(recs[i].h);
                for (size_t k = i; k < j; k++) hrec.push_back(recs[k].r);
                hptr.push_back((int32_t)hrec.size());
                i = j;
            }
        }
        hkeys.shrink_to_fit(); hptr.shrink_to_fit(); hrec.shrink_to_fit();
        hfinalized = true;
    }

    // binary search the sorted string key arena → key index, or -1
    int find_key(const std::string & key) const {
        int lo = 0, hi = (int)key_off.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            std::string_view cur(key_blob.data() + key_off[mid], (size_t)(key_off[mid+1] - key_off[mid]));
            int c = cur.compare(std::string_view(key));
            if (c == 0) return mid;
            if (c < 0) lo = mid + 1; else hi = mid;
        }
        return -1;
    }

    std::vector<record_entry> lookup(const std::string & key) const {
        if (hfinalized) {   // hash-keyed streaming CSR
            uint64_t h = hashkey(key);
            auto lo = std::lower_bound(hkeys.begin(), hkeys.end(), h);
            if (lo == hkeys.end() || *lo != h) return {};
            int i = (int)(lo - hkeys.begin());
            return std::vector<record_entry>(hrec.begin() + hptr[i], hrec.begin() + hptr[i+1]);
        }
        if (finalized) {    // string-keyed CSR
            int i = find_key(key);
            if (i < 0) return {};
            return std::vector<record_entry>(rec_data.begin() + rec_ptr[i], rec_data.begin() + rec_ptr[i+1]);
        }
        auto it = posting.find(key);
        if (it != posting.end()) return it->second;
        return {};
    }

    std::vector<record_entry> lookup_any(const std::vector<std::string> & keys) const {
        std::vector<record_entry> result;
        std::unordered_set<int64_t> seen;
        for (auto & k : keys) {
            auto recs = lookup(k);
            for (auto & rec : recs) {
                int64_t key_id = (int64_t)rec.chunk_id * 1000000 + rec.byte_offset;
                if (seen.insert(key_id).second) result.push_back(rec);
            }
        }
        return result;
    }

    // selfcheck: streamed lookup(key) must equal the map's records (as a set) for every key.
    void run_selfcheck(const char * name) const {
        int ok = 0, bad = 0;
        for (auto & kv : posting) {
            auto got = lookup(kv.first);
            std::unordered_set<int64_t> a, b;
            for (auto & r : kv.second) a.insert((int64_t)r.chunk_id * 1000000 + r.byte_offset);
            for (auto & r : got)       b.insert((int64_t)r.chunk_id * 1000000 + r.byte_offset);
            if (a == b) ok++; else bad++;
        }
        fprintf(stderr, "[meta-selfcheck %s] keys=%d MATCH=%d MISMATCH=%d\n", name, (int)posting.size(), ok, bad);
    }

    int n_keys() const { return hfinalized ? (int)hkeys.size() : (finalized ? (int)key_off.size() - 1 : (int)posting.size()); }
};

// Intersect two record lists by (chunk_id, byte_offset)
static std::vector<record_entry> intersect_records(
    const std::vector<record_entry> & a,
    const std::vector<record_entry> & b)
{
    std::unordered_set<int64_t> b_set;
    for (auto & r : b) {
        b_set.insert((int64_t)r.chunk_id * 1000000 + r.byte_offset);
    }
    std::vector<record_entry> result;
    for (auto & r : a) {
        if (b_set.count((int64_t)r.chunk_id * 1000000 + r.byte_offset)) {
            result.push_back(r);
        }
    }
    return result;
}

// Metadata extraction (auto-detect from any text)

static std::string to_lower(const std::string & s) {
    std::string r = s;
    for (auto & c : r) c = (char)std::tolower((unsigned char)c);
    return r;
}

// US state names + abbreviations for location detection
static const std::unordered_map<std::string, std::string> & get_us_states() {
    static const std::unordered_map<std::string, std::string> states = {
        {"alabama","AL"},{"alaska","AK"},{"arizona","AZ"},{"arkansas","AR"},
        {"california","CA"},{"colorado","CO"},{"connecticut","CT"},{"delaware","DE"},
        {"florida","FL"},{"georgia","GA"},{"hawaii","HI"},{"idaho","ID"},
        {"illinois","IL"},{"indiana","IN"},{"iowa","IA"},{"kansas","KS"},
        {"kentucky","KY"},{"louisiana","LA"},{"maine","ME"},{"maryland","MD"},
        {"massachusetts","MA"},{"michigan","MI"},{"minnesota","MN"},{"mississippi","MS"},
        {"missouri","MO"},{"montana","MT"},{"nebraska","NE"},{"nevada","NV"},
        {"new hampshire","NH"},{"new jersey","NJ"},{"new mexico","NM"},{"new york","NY"},
        {"north carolina","NC"},{"north dakota","ND"},{"ohio","OH"},{"oklahoma","OK"},
        {"oregon","OR"},{"pennsylvania","PA"},{"rhode island","RI"},{"south carolina","SC"},
        {"south dakota","SD"},{"tennessee","TN"},{"texas","TX"},{"utah","UT"},
        {"vermont","VT"},{"virginia","VA"},{"washington","WA"},{"west virginia","WV"},
        {"wisconsin","WI"},{"wyoming","WY"},
        // Abbreviation -> abbreviation (for reverse lookup)
        {"al","AL"},{"ak","AK"},{"az","AZ"},{"ar","AR"},{"ca","CA"},{"co","CO"},
        {"ct","CT"},{"de","DE"},{"fl","FL"},{"ga","GA"},{"hi","HI"},{"id","ID"},
        {"il","IL"},{"in","IN"},{"ia","IA"},{"ks","KS"},{"ky","KY"},{"la","LA"},
        {"me","ME"},{"md","MD"},{"ma","MA"},{"mi","MI"},{"mn","MN"},{"ms","MS"},
        {"mo","MO"},{"mt","MT"},{"ne","NE"},{"nv","NV"},{"nh","NH"},{"nj","NJ"},
        {"nm","NM"},{"ny","NY"},{"nc","NC"},{"nd","ND"},{"oh","OH"},{"ok","OK"},
        {"or","OR"},{"pa","PA"},{"ri","RI"},{"sc","SC"},{"sd","SD"},{"tn","TN"},
        {"tx","TX"},{"ut","UT"},{"vt","VT"},{"va","VA"},{"wa","WA"},{"wv","WV"},
        {"wi","WI"},{"wy","WY"},
    };
    return states;
}

// Extract person names: capitalized word pairs
static std::vector<std::string> extract_persons(const std::string & text) {
    std::vector<std::string> persons;
    static const std::regex re_person(R"(([A-Z][a-z]+)\s+([A-Z][a-z]+))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_person);
    auto end = std::sregex_iterator();
    std::unordered_set<std::string> seen;
    for (auto it = begin; it != end; ++it) {
        std::string name = to_lower((*it)[1].str() + " " + (*it)[2].str());
        // Skip common non-name pairs
        static const std::unordered_set<std::string> skip = {
            "new york", "new jersey", "new mexico", "new hampshire",
            "north carolina", "north dakota", "south carolina", "south dakota",
            "west virginia", "rhode island", "los angeles", "san francisco",
            "las vegas", "el paso", "des moines",
        };
        if (!skip.count(name) && seen.insert(name).second) {
            persons.push_back(name);
        }
    }
    return persons;
}

// Expand person name to aliases
static std::vector<std::string> expand_person_aliases(const std::string & name) {
    std::vector<std::string> aliases;
    aliases.push_back(name);
    auto space = name.find(' ');
    if (space != std::string::npos) {
        std::string first = name.substr(0, space);
        std::string last = name.substr(space + 1);
        aliases.push_back(last + ", " + first);           // anderson, peter
        aliases.push_back(last + " " + first);            // anderson peter
        aliases.push_back(first.substr(0, 1) + ". " + last); // p. anderson
        aliases.push_back(first.substr(0, 1) + " " + last);  // p anderson
    }
    return aliases;
}

// Extract dates: years, YYYY-MM-DD, YYYY-MM, month names
static std::vector<std::pair<std::string, double>> extract_dates(const std::string & text) {
    std::vector<std::pair<std::string, double>> dates;
    // YYYY-MM-DD
    static const std::regex re_full(R"(\b(\d{4})[-/](\d{2})[-/](\d{2})\b)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_full);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        int y = std::stoi((*it)[1].str());
        int m = std::stoi((*it)[2].str());
        int d = std::stoi((*it)[3].str());
        if (y >= 1900 && y <= 2100 && m >= 1 && m <= 12 && d >= 1 && d <= 31) {
            double ts = y * 10000 + m * 100 + d;  // sortable numeric: 20250115
            dates.push_back({(*it)[0].str(), ts});
        }
    }
    // YYYY-MM
    static const std::regex re_ym(R"(\b(\d{4})[-/](\d{2})\b)");
    begin = std::sregex_iterator(text.begin(), text.end(), re_ym);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        int y = std::stoi((*it)[1].str());
        int m = std::stoi((*it)[2].str());
        if (y >= 1900 && y <= 2100 && m >= 1 && m <= 12) {
            double ts = y * 10000 + m * 100;
            dates.push_back({(*it)[0].str(), ts});
        }
    }
    // Standalone years
    static const std::regex re_year(R"(\b(19|20)\d{2}\b)");
    begin = std::sregex_iterator(text.begin(), text.end(), re_year);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        int y = std::stoi((*it)[0].str());
        double ts = y * 10000;
        dates.push_back({(*it)[0].str(), ts});
    }
    return dates;
}

// Extract locations: US states + major cities
static std::vector<std::string> extract_locations(const std::string & text) {
    std::vector<std::string> locs;
    std::string lower = to_lower(text);
    auto & states = get_us_states();
    std::unordered_set<std::string> seen;
    // Check for state names and abbreviations
    for (auto & kv : states) {
        if (kv.first.size() <= 2) continue;  // skip abbreviation keys
        if (lower.find(kv.first) != std::string::npos) {
            if (seen.insert(kv.second).second) {
                locs.push_back(kv.second);
            }
        }
    }
    // Check for 2-letter state abbreviations (uppercase in original text)
    static const std::regex re_abbr(R"(\b([A-Z]{2})\b)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_abbr);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        std::string abbr = to_lower((*it)[1].str());
        auto st = states.find(abbr);
        if (st != states.end() && seen.insert(st->second).second) {
            locs.push_back(st->second);
        }
    }
    return locs;
}

// Parse a double, returning false on malformed input instead of throwing. Real prose feeds
// these regexes tokens like "$," (empty after stripping) or overflowing digit runs that
// std::stod rejects; a value that won't parse is dropped rather than aborting the ingest.
static bool safe_stod(const std::string & s, double & out) {
    if (s.empty()) return false;
    try { size_t pos = 0; out = std::stod(s, &pos); return pos > 0; }
    catch (...) { return false; }
}

// Extract numeric values with optional units
static std::vector<std::pair<std::string, double>> extract_numerics(const std::string & text) {
    std::vector<std::pair<std::string, double>> nums;
    // Currency: $1,234.56 or $10K
    static const std::regex re_money(R"(\$[\d,]+\.?\d*[KMB]?)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_money);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        std::string s = (*it)[0].str().substr(1);  // strip $
        // Remove commas
        s.erase(std::remove(s.begin(), s.end(), ','), s.end());
        if (s.empty()) continue;
        double v = 0, base = 0;
        char suffix = s.back();
        if (suffix == 'K' || suffix == 'k') { s.pop_back(); if (!safe_stod(s, base)) continue; v = base * 1000; }
        else if (suffix == 'M' || suffix == 'm') { s.pop_back(); if (!safe_stod(s, base)) continue; v = base * 1000000; }
        else if (suffix == 'B' || suffix == 'b') { s.pop_back(); if (!safe_stod(s, base)) continue; v = base * 1000000000; }
        else { if (!safe_stod(s, v)) continue; }
        nums.push_back({(*it)[0].str(), v});
    }
    // Temperature: 100°F, 37.5°C
    static const std::regex re_temp(R"((\d+\.?\d*)\s*°\s*([FC]))");
    begin = std::sregex_iterator(text.begin(), text.end(), re_temp);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        double v = 0;
        if (!safe_stod((*it)[1].str(), v)) continue;
        nums.push_back({(*it)[0].str(), v});
    }
    // Percentage: 95.5%
    static const std::regex re_pct(R"((\d+\.?\d*)\s*%)");
    begin = std::sregex_iterator(text.begin(), text.end(), re_pct);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        double v = 0;
        if (!safe_stod((*it)[1].str(), v)) continue;
        nums.push_back({(*it)[0].str(), v});
    }
    return nums;
}

// Extract categories: field labels like "Type: insurance", "Status: pending"
static std::vector<std::pair<std::string, std::string>> extract_categories(const std::string & text) {
    std::vector<std::pair<std::string, std::string>> cats;
    static const std::regex re_label(R"((?:type|status|category|class|kind|record|action)\s*[:=]\s*(\w+))", std::regex::icase);
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_label);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        cats.push_back({"type", to_lower((*it)[1].str())});
    }
    // Domain keywords as implicit categories
    static const std::vector<std::pair<std::string, std::string>> keyword_cats = {
        {"insurance", "insurance"}, {"claim", "claim"}, {"fraud", "fraud"},
        {"traffic", "traffic"}, {"violation", "violation"}, {"dui", "dui"},
        {"accident", "accident"}, {"prescription", "medical"}, {"diagnosis", "medical"},
        {"invoice", "financial"}, {"transaction", "financial"}, {"payment", "financial"},
    };
    std::string lower = to_lower(text);
    for (auto & kc : keyword_cats) {
        if (lower.find(kc.first) != std::string::npos) {
            cats.push_back({"category", kc.second});
        }
    }
    return cats;
}

// Extract IDs: case #4521, patient ID P-789, account 12345
static std::vector<std::pair<std::string, std::string>> extract_ids(const std::string & text) {
    std::vector<std::pair<std::string, std::string>> ids;
    static const std::regex re_id(R"((?:case|patient|account|order|sensor|claim|policy|record|ticket|id)\s*[#:]?\s*([\w-]+))", std::regex::icase);
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_id);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        ids.push_back({"id", to_lower((*it)[1].str())});
    }
    return ids;
}

// Full metadata index

struct metadata_index {
    inverted_index person_idx;
    range_index    date_idx;
    inverted_index location_idx;
    range_index    numeric_idx;
    inverted_index category_idx;
    inverted_index id_idx;

    int n_records = 0;
    bool streaming = false;

    // Batch mode: stream the four inverted indexes' postings to temp files instead of holding
    // resident maps. Range indexes stay in RAM (they are small: interned value + record_entry).
    void enable_streaming(const std::string & prefix, bool selfcheck) {
        streaming = true;
        person_idx.open_stream(prefix + "_person.stream", selfcheck);
        location_idx.open_stream(prefix + "_location.stream", selfcheck);
        category_idx.open_stream(prefix + "_category.stream", selfcheck);
        id_idx.open_stream(prefix + "_id.stream", selfcheck);
    }

    // Pack the four inverted indexes (sorted arena/CSR, or hash-keyed CSR if streaming) after ingest.
    void finalize() {
        person_idx.finalize();
        location_idx.finalize();
        category_idx.finalize();
        id_idx.finalize();
        if (streaming && (person_idx.selfcheck)) {
            person_idx.run_selfcheck("person");
            location_idx.run_selfcheck("location");
            category_idx.run_selfcheck("category");
            id_idx.run_selfcheck("id");
        }
        date_idx.ensure_sorted();     // range indexes are already vectors; just sort once
        numeric_idx.ensure_sorted();
    }

    // Resident-RAM estimate (bytes) of the six sub-indexes' postings, for RSS profiling.
    size_t resident_bytes() const {
        auto inv = [](const inverted_index & ix) -> size_t {
            if (ix.hfinalized)
                return ix.hkeys.capacity()*8 + ix.hptr.capacity()*4
                     + ix.hrec.capacity()*sizeof(record_entry);
            if (ix.finalized)
                return ix.key_blob.capacity() + ix.key_off.capacity()*4
                     + ix.rec_ptr.capacity()*4 + ix.rec_data.capacity()*sizeof(record_entry);
            size_t b = ix.posting.bucket_count()*8;
            for (const auto & p : ix.posting)
                b += 80 + p.first.capacity() + p.second.capacity()*sizeof(record_entry);
            return b;
        };
        auto rng = [](const range_index & ix) -> size_t {
            return ix.entries.capacity()*sizeof(range_index::entry);
        };
        return inv(person_idx) + inv(location_idx) + inv(category_idx) + inv(id_idx)
             + rng(date_idx) + rng(numeric_idx);
    }

    // Index a single record's metadata
    void add_record(const record_entry & rec, const std::string & record_text) {
        // Person names + aliases
        auto persons = extract_persons(record_text);
        for (auto & p : persons) {
            auto aliases = expand_person_aliases(p);
            for (auto & a : aliases) {
                person_idx.add(a, rec);
            }
        }

        // Dates
        auto dates = extract_dates(record_text);
        for (auto & d : dates) {
            date_idx.add(d.second, rec);
        }

        // Locations
        auto locs = extract_locations(record_text);
        for (auto & l : locs) {
            location_idx.add(l, rec);
        }

        // Numerics
        auto nums = extract_numerics(record_text);
        for (auto & n : nums) {
            numeric_idx.add(n.second, rec);
        }

        // Categories
        auto cats = extract_categories(record_text);
        for (auto & c : cats) {
            category_idx.add(c.second, rec);
        }

        // IDs
        auto ids = extract_ids(record_text);
        for (auto & id : ids) {
            id_idx.add(id.second, rec);
        }

        n_records++;
    }

    // Parse a chunk into records and index each one
    void index_chunk(int chunk_id, const std::string & text) {
        // Detect record boundaries: blank lines, or fixed-size records
        std::vector<std::pair<int, int>> boundaries;  // (offset, length)

        int start = 0;
        int len = (int)text.size();

        // Split on double newlines (paragraph boundaries)
        for (int i = 0; i < len; i++) {
            if (i + 1 < len && text[i] == '\n' && text[i + 1] == '\n') {
                if (i - start > 20) {  // minimum record size
                    boundaries.push_back({start, i - start});
                }
                start = i + 2;
            }
        }
        if (len - start > 20) {
            boundaries.push_back({start, len - start});
        }

        // If no paragraph breaks found, treat whole chunk as one record
        if (boundaries.empty()) {
            boundaries.push_back({0, len});
        }

        for (auto & b : boundaries) {
            record_entry rec = {chunk_id, b.first, b.second};
            std::string record_text = text.substr(b.first, b.second);
            add_record(rec, record_text);
        }
    }

    // Query: decompose query string -> intersect indexes
    struct query_result {
        std::vector<record_entry> records;
        int n_constraints = 0;
        bool used_metadata = false;
    };

    query_result query(const std::string & query_text) const {
        query_result qr;

        // Decompose query into constraints
        auto persons = extract_persons(query_text);
        auto dates = extract_dates(query_text);
        auto locs = extract_locations(query_text);
        auto cats = extract_categories(query_text);
        auto ids = extract_ids(query_text);

        // Also check for numeric range patterns: ">$10K", "above 100"
        // (simplified: extract numerics from query for range matching)
        auto nums = extract_numerics(query_text);

        // Collect constraint results
        std::vector<std::vector<record_entry>> constraint_results;

        // Person constraint
        if (!persons.empty()) {
            std::vector<std::string> all_aliases;
            for (auto & p : persons) {
                auto aliases = expand_person_aliases(p);
                all_aliases.insert(all_aliases.end(), aliases.begin(), aliases.end());
            }
            auto hits = person_idx.lookup_any(all_aliases);
            if (!hits.empty()) {
                constraint_results.push_back(hits);
                qr.n_constraints++;
            }
        }

        // Date constraint (use year range)
        if (!dates.empty()) {
            double min_date = dates[0].second;
            double max_date = dates[0].second;
            for (auto & d : dates) {
                min_date = std::min(min_date, d.second);
                max_date = std::max(max_date, d.second);
            }
            // Expand year to full year range: 20250000 -> 20250000-20251231
            if (std::fmod(max_date, 10000) == 0) max_date += 1231;
            auto hits = const_cast<range_index &>(date_idx).range(min_date, max_date);
            if (!hits.empty()) {
                constraint_results.push_back(hits);
                qr.n_constraints++;
            }
        }

        // Location constraint
        if (!locs.empty()) {
            std::vector<record_entry> loc_hits;
            for (auto & l : locs) {
                auto h = location_idx.lookup(l);
                loc_hits.insert(loc_hits.end(), h.begin(), h.end());
            }
            if (!loc_hits.empty()) {
                constraint_results.push_back(loc_hits);
                qr.n_constraints++;
            }
        }

        // Category constraint
        if (!cats.empty()) {
            std::vector<record_entry> cat_hits;
            for (auto & c : cats) {
                auto h = category_idx.lookup(c.second);
                cat_hits.insert(cat_hits.end(), h.begin(), h.end());
            }
            if (!cat_hits.empty()) {
                constraint_results.push_back(cat_hits);
                qr.n_constraints++;
            }
        }

        // ID constraint
        if (!ids.empty()) {
            for (auto & id : ids) {
                auto h = id_idx.lookup(id.second);
                if (!h.empty()) {
                    constraint_results.push_back(h);
                    qr.n_constraints++;
                }
            }
        }

        // Intersect all constraints
        if (constraint_results.empty()) {
            qr.used_metadata = false;
            return qr;
        }

        qr.used_metadata = true;

        // Start with smallest set (most selective), intersect with others
        std::sort(constraint_results.begin(), constraint_results.end(),
            [](auto & a, auto & b) { return a.size() < b.size(); });

        qr.records = constraint_results[0];
        for (size_t i = 1; i < constraint_results.size() && !qr.records.empty(); i++) {
            qr.records = intersect_records(qr.records, constraint_results[i]);
        }

        return qr;
    }
};
