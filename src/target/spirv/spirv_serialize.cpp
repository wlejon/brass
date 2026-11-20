#include <brass/target/spirv_target.hpp>

#include <cstring>
#include <limits>

namespace brass::target {
namespace {

constexpr uint8_t kMagic[4] = {'B', 'S', 'P', 'K'};
constexpr uint32_t kFormatVersion = 1;

void write_u8(std::vector<uint8_t>& out, uint8_t v) {
    out.push_back(v);
}

void write_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
    }
}

void write_str(std::vector<uint8_t>& out, const std::string& s) {
    write_u32(out, static_cast<uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

class Reader {
public:
    explicit Reader(std::span<const uint8_t> data, std::string* err)
        : data_(data), err_(err) {}

    bool set_err(const char* msg) {
        if (ok_) {
            ok_ = false;
            if (err_) *err_ = msg;
        }
        return false;
    }

    size_t remaining() const noexcept { return data_.size() - pos_; }
    bool ok() const noexcept { return ok_; }
    bool at_end() const noexcept { return ok_ && pos_ == data_.size(); }

    bool check_bytes(const uint8_t* expected, size_t n, const char* msg) {
        if (!ok_ || remaining() < n) return set_err(msg);
        if (std::memcmp(data_.data() + pos_, expected, n) != 0) return set_err(msg);
        pos_ += n;
        return true;
    }

    uint8_t u8() {
        if (!ok_ || remaining() < 1) {
            set_err("truncated SpirvKernel stream (u8)");
            return 0;
        }
        return data_[pos_++];
    }

    bool boolean() {
        uint8_t v = u8();
        if (v > 1) {
            set_err("invalid boolean byte in SpirvKernel stream");
            return false;
        }
        return v != 0;
    }

    uint32_t u32() {
        if (!ok_ || remaining() < 4) {
            set_err("truncated SpirvKernel stream (u32)");
            return 0;
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(data_[pos_ + static_cast<size_t>(i)]) << (8 * i);
        }
        pos_ += 4;
        return v;
    }

    size_t count(size_t elem_min_bytes) {
        uint32_t n = u32();
        if (!ok_) return 0;
        if (elem_min_bytes > 0 && static_cast<size_t>(n) > remaining() / elem_min_bytes) {
            set_err("SpirvKernel count exceeds remaining payload");
            return 0;
        }
        return static_cast<size_t>(n);
    }

    std::string str() {
        size_t len = count(1);
        if (!ok_) return {};
        std::string s(reinterpret_cast<const char*>(data_.data() + pos_), len);
        pos_ += len;
        return s;
    }

private:
    std::span<const uint8_t> data_;
    size_t pos_ = 0;
    std::string* err_ = nullptr;
    bool ok_ = true;
};

} // namespace

std::vector<uint8_t> serialize_spirv_kernel(const SpirvKernel& kernel) {
    std::vector<uint8_t> out;
    out.reserve(64 + kernel.words.size() * 4 + kernel.params.size() * 9 + kernel.descriptors.size() * 25);

    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
    write_u32(out, kFormatVersion);

    write_str(out, kernel.entry);
    write_u8(out, static_cast<uint8_t>(kernel.stage));

    write_u32(out, static_cast<uint32_t>(kernel.words.size()));
    for (uint32_t w : kernel.words) {
        write_u32(out, w);
    }

    write_u32(out, static_cast<uint32_t>(kernel.params.size()));
    for (const SpirvParam& p : kernel.params) {
        write_u8(out, static_cast<uint8_t>(p.type.kind()));
        write_u32(out, p.offset);
        write_u32(out, p.size);
    }

    write_u32(out, static_cast<uint32_t>(kernel.descriptors.size()));
    for (const SpirvDescriptorBinding& d : kernel.descriptors) {
        write_u32(out, d.set);
        write_u32(out, d.binding);
        write_u8(out, static_cast<uint8_t>(d.kind));
        write_u8(out, static_cast<uint8_t>(d.dim));
        write_u8(out, d.arrayed ? 1u : 0u);
        write_u8(out, d.depth ? 1u : 0u);
        write_u8(out, static_cast<uint8_t>(d.scalar));
        write_u32(out, d.format);
        write_u32(out, d.bytes);
    }

    write_u32(out, kernel.push_constant_bytes);
    write_u32(out, kernel.shared_bytes);
    for (uint32_t l : kernel.local_size) {
        write_u32(out, l);
    }
    write_u8(out, kernel.local_size_spec_constants ? 1u : 0u);

    write_u32(out, static_cast<uint32_t>(kernel.capabilities.size()));
    for (const std::string& cap : kernel.capabilities) {
        write_str(out, cap);
    }

    write_u32(out, static_cast<uint32_t>(kernel.extensions.size()));
    for (const std::string& ext : kernel.extensions) {
        write_str(out, ext);
    }

    return out;
}

bool deserialize_spirv_kernel(std::span<const uint8_t> bytes,
                              SpirvKernel& out_kernel,
                              std::string* err_out) {
    Reader r(bytes, err_out);
    if (!r.check_bytes(kMagic, sizeof(kMagic), "invalid SpirvKernel magic header")) {
        return false;
    }
    uint32_t version = r.u32();
    if (!r.ok() || version != kFormatVersion) {
        r.set_err("unsupported SpirvKernel format version");
        return false;
    }

    SpirvKernel parsed;
    parsed.entry = r.str();
    uint8_t stage_u8 = r.u8();
    if (stage_u8 > static_cast<uint8_t>(SpirvStage::Fragment)) {
        r.set_err("invalid SpirvStage value in SpirvKernel stream");
        return false;
    }
    parsed.stage = static_cast<SpirvStage>(stage_u8);

    size_t nwords = r.count(4);
    parsed.words.resize(nwords);
    for (size_t i = 0; i < nwords; ++i) {
        parsed.words[i] = r.u32();
    }

    size_t nparams = r.count(9);
    parsed.params.resize(nparams);
    for (size_t i = 0; i < nparams; ++i) {
        uint8_t tk = r.u8();
        if (tk > static_cast<uint8_t>(TypeKind::Tagged)) {
            r.set_err("invalid TypeKind in SpirvKernel params");
            return false;
        }
        parsed.params[i].type = Type(static_cast<TypeKind>(tk));
        parsed.params[i].offset = r.u32();
        parsed.params[i].size = r.u32();
    }

    size_t ndescs = r.count(21);
    parsed.descriptors.resize(ndescs);
    for (size_t i = 0; i < ndescs; ++i) {
        SpirvDescriptorBinding& d = parsed.descriptors[i];
        d.set = r.u32();
        d.binding = r.u32();
        uint8_t kind = r.u8();
        uint8_t dim = r.u8();
        d.arrayed = r.boolean();
        d.depth = r.boolean();
        uint8_t scalar = r.u8();
        if (kind > static_cast<uint8_t>(SpirvDescriptorKind::StorageBuffer) ||
            dim > static_cast<uint8_t>(SpirvImageDim::Cube) ||
            scalar > static_cast<uint8_t>(SpirvImageScalar::I32)) {
            r.set_err("invalid descriptor enum in SpirvKernel stream");
            return false;
        }
        d.kind = static_cast<SpirvDescriptorKind>(kind);
        d.dim = static_cast<SpirvImageDim>(dim);
        d.scalar = static_cast<SpirvImageScalar>(scalar);
        d.format = r.u32();
        d.bytes = r.u32();
    }

    parsed.push_constant_bytes = r.u32();
    parsed.shared_bytes = r.u32();
    for (uint32_t& l : parsed.local_size) {
        l = r.u32();
    }
    parsed.local_size_spec_constants = r.boolean();

    size_t ncaps = r.count(4);
    parsed.capabilities.resize(ncaps);
    for (size_t i = 0; i < ncaps; ++i) {
        parsed.capabilities[i] = r.str();
    }

    size_t nexts = r.count(4);
    parsed.extensions.resize(nexts);
    for (size_t i = 0; i < nexts; ++i) {
        parsed.extensions[i] = r.str();
    }

    if (!r.at_end()) {
        r.set_err("trailing bytes after SpirvKernel payload");
        return false;
    }

    out_kernel = std::move(parsed);
    return true;
}

} // namespace brass::target
