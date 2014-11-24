
/// <reference path="../../ref.ts" />

import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import utils = require("./../../utils/utils");

class Barrier {
    // you can represent this as a typed UInt8Array,
    // but it does not make a difference since you'd
    // be optimizing for one browser (see https://github.com/sq/JSIL/issues/250)
    public mask : Array<boolean>;
    constructor(dim : utils.Dim3) {
        mask = new Array(dim.flattendLength());
    }
}

export = Barrier
