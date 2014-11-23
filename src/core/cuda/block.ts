
/// <reference path="../../ref.ts" />

import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import utils = require("./../../utils/utils");

class Block {
    public blockIdx : utils.Dim3 = new utils.Dim3(0);
    public blockDim : utils.Dim3 = new utils.Dim3(0);
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public threads : thread[] = null;
}

export = Block
