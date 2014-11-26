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
        var utils;
        (function (utils) {
            utils.logger = new Logger_();
            utils.assert = assert_;
            utils.guuid = guuid_;
            utils.rand = rand_;
            utils.castTo = castTo_;
            function applyMixins(derivedCtor, baseCtors) {
                baseCtors.forEach(function (baseCtor) {
                    Object.getOwnPropertyNames(baseCtor.prototype).forEach(function (name) {
                        derivedCtor.prototype[name] = baseCtor.prototype[name];
                    });
                });
            }
            utils.applyMixins = applyMixins;
        })(utils = lib.utils || (lib.utils = {}));
    })(lib || (lib = {}));
});
