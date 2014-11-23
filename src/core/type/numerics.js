define(["require", "exports"], function (require, exports) {
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
});
//# sourceMappingURL=numerics.js.map