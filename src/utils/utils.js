define(["require", "exports"], function (require, exports) {
    /// <reference path="../ref.ts" />
    var Core;
    (function (Core) {
        Core.guuid = function () {
            var s4 = function () { return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1); };
            return s4() + s4() + "-" + s4() + "-" + s4() + "-" + s4() + "-" + s4() + s4() + s4();
        };
        var Dim3 = (function () {
            function Dim3(x, y, z) {
                if (y === void 0) { y = 1; }
                if (z === void 0) { z = 1; }
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