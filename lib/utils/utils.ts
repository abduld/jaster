
/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />
/// <reference path="./castTo.ts" />


module lib {

    export module utils {
        export var logger = new lib.utils.detail.Logger();
        export import assert = lib.utils.detail.assert;
        export import guuid = lib.utils.detail.guuid;
        export import rand = lib.utils.detail.rand;
        export import castTo = lib.utils.detail.castTo;

        export function applyMixins(derivedCtor: any, baseCtors: any[]) {
            baseCtors.forEach(baseCtor => {
                Object.getOwnPropertyNames(baseCtor.prototype).forEach(name => {
                    derivedCtor.prototype[name] = baseCtor.prototype[name];
                })
            });
        }
    }
}
