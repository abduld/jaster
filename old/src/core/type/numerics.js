define(["require", "exports", "./int8", "./uint8", "./int16", "./uint16", "./int32", "./uint32", "./int64"], function(require, exports, int8_, uint8_, int16_, uint16_, int32_, uint32_, int64_) {
    /// <reference path="../../ref.ts" />
    (function (CNumberKind) {
        CNumberKind[CNumberKind["Int8"] = 10] = "Int8";
        CNumberKind[CNumberKind["Uint8"] = 11] = "Uint8";
        CNumberKind[CNumberKind["Int16"] = 20] = "Int16";
        CNumberKind[CNumberKind["Uint16"] = 21] = "Uint16";
        CNumberKind[CNumberKind["Int32"] = 30] = "Int32";
        CNumberKind[CNumberKind["Uint32"] = 31] = "Uint32";
        CNumberKind[CNumberKind["Int64"] = 40] = "Int64";
        CNumberKind[CNumberKind["Float"] = 52] = "Float";
        CNumberKind[CNumberKind["Double"] = 62] = "Double";
    })(exports.CNumberKind || (exports.CNumberKind = {}));
    var CNumberKind = exports.CNumberKind;

    exports.CNumberKindMap = null;
    if (exports.CNumberKindMap === null) {
        exports.CNumberKindMap = new Map();
    }

    exports.int8 = int8_;
    exports.uint8 = uint8_;
    exports.int16 = int16_;
    exports.uint16 = uint16_;
    exports.int32 = int32_;
    exports.uint32 = uint32_;
    exports.int64 = int64_;
});
//# sourceMappingURL=numerics.js.map
