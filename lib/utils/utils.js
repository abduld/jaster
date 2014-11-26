/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />
/// <reference path="./castTo.ts" />
define(["require", "exports", "./logger", "./assert", "./guuid", "./rand", "./castTo"], function (require, exports, Logger_, assert_, guuid_, rand_, castTo_) {
    var lib;
    (function (lib) {
        var detail;
        (function (detail) {
            var Utils = (function () {
                function Utils() {
                    this.logger = new Logger_();
                    this.assert = assert_;
                    this.guuid = guuid_;
                    this.rand = rand_;
                    this.castTo = castTo_;
                }
                Utils.prototype.applyMixins = function (derivedCtor, baseCtors) {
                    baseCtors.forEach(function (baseCtor) {
                        Object.getOwnPropertyNames(baseCtor.prototype).forEach(function (name) {
                            derivedCtor.prototype[name] = baseCtor.prototype[name];
                        });
                    });
                };
                return Utils;
            })();
            detail.Utils = Utils;
            detail.utils = new Utils();
        })(detail || (detail = {}));
        lib.utils = detail.utils;
    })(lib || (lib = {}));
});
