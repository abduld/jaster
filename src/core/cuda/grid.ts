
/// <reference path="../../ref.ts" />

import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import utils = require("./../../utils/utils");

class Grid {
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public blocks : block[] = null;
}

export = Grid