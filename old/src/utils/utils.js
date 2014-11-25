define(["require", "exports"], function(require, exports) {
    /// <reference path="../ref.ts" />
    var Core;
    (function (Core) {
        (function (ErrorCode) {
            ErrorCode[ErrorCode["Success"] = 0] = "Success";
            ErrorCode[ErrorCode["MemoryOverflow"] = 1] = "MemoryOverflow";
            ErrorCode[ErrorCode["IntegerOverflow"] = 2] = "IntegerOverflow";
            ErrorCode[ErrorCode["Unknown"] = 3] = "Unknown";
        })(Core.ErrorCode || (Core.ErrorCode = {}));
        var ErrorCode = Core.ErrorCode;
        ;
        var Error = (function () {
            function Error(code) {
                if (code) {
                    this.code = code;
                } else {
                    this.code = 0 /* Success */;
                }
            }
            return Error;
        })();
        Core.Error = Error;
        Core.guuid = function () {
            var s4 = function () {
                return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
            };
            return s4() + s4() + "-" + s4() + "-" + s4() + "-" + s4() + "-" + s4() + s4() + s4();
        };

        var Dim3 = (function () {
            function Dim3(x, y, z) {
                if (typeof y === "undefined") { y = 1; }
                if (typeof z === "undefined") { z = 1; }
                this.x = x;
                this.y = y;
                this.z = z;
            }
            return Dim3;
        })();
        Core.Dim3 = Dim3;

        Core.rand = function (min, max) {
            return min + Math.random() * (max - min);
        };
        function applyMixins(derivedCtor, baseCtors) {
            baseCtors.forEach(function (baseCtor) {
                Object.getOwnPropertyNames(baseCtor.prototype).forEach(function (name) {
                    derivedCtor.prototype[name] = baseCtor.prototype[name];
                });
            });
        }
        Core.applyMixins = applyMixins;
    })(Core || (Core = {}));
    
    return Core;
});
//# sourceMappingURL=utils.js.map
