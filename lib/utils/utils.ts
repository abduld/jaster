
/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />

import Logger_ = require("./logger");
import Assert_ = require("./assert");
import Guuid_ = require("./guuid");

class Utils {
    Logger = Logger_;
    Assert = Assert_;
    Guuid = Guuid_;
};

var utils = new Utils();
export = utils;

