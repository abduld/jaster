/// <reference path="../ref.ts" />
define(["require", "exports", "./type/numerics", "./mem/memory", "./type/int8", "./type/uint8", "./type/int16", "./type/uint16", "./type/int32", "./type/uint32", "./type/int64"], function(require, exports, numerics_, memory_, int8_, uint8_, int16_, uint16_, int32_, uint32_, int64_) {
    exports.VERSION = 0.1;
    exports.numerics = numerics_;
    exports.int8 = int8_;
    exports.uint8 = uint8_;
    exports.int16 = int16_;
    exports.uint16 = uint16_;
    exports.int32 = int32_;
    exports.uint32 = uint32_;
    exports.int64 = int64_;
    exports.memory = memory_;
});
//# sourceMappingURL=core.js.map
