
/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />
/// <reference path="./castTo.ts" />

import Logger_ = require("./logger");
import assert_ = require("./assert");
import guuid_ = require("./guuid");
import rand_ = require("./rand");
import castTo_ = require("./castTo");

module lib {
    module detail {
        export class Utils {
            logger = new Logger_();
            assert = assert_;
            guuid = guuid_;
            rand = rand_;
            castTo = castTo_;

            applyMixins(derivedCtor: any, baseCtors: any[]) {
                baseCtors.forEach(baseCtor => {
                    Object.getOwnPropertyNames(baseCtor.prototype).forEach(name => {
                        derivedCtor.prototype[name] = baseCtor.prototype[name];
                    })
                });
            }
        }
        export var utils = new Utils();
    }
    export import utils = detail.utils;
}
