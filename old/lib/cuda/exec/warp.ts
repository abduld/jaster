
/// <reference path="../../ref.ts" />


import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import utils = require("./../../utils/utils");

class Warp {
    public id : string = utils.guuid();
    public thread : thread = null;
}

export = Warp