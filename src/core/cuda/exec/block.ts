
/// <reference path="../../ref.ts" />

import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import barrier = require("./barrier");
import utils = require("./../../utils/utils");

class Block {
    public blockIdx : utils.Dim3 = new utils.Dim3(0);
    public blockDim : utils.Dim3 = new utils.Dim3(0);
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public threads : thread[] = null;
    public barriers : barrier[] = null;
    constructor(block : block, threadIdx : utils.Dim3, fun : Function, args : Array<any>) {
        this.status = cuda.Status.Idle;
        this.block = block;
        this.blockIdx = block.blockIdx;
        this.gridIdx = block.gridIdx;
        this.gridDim = block.gridDim;
        this.threadIdx = threadIdx;
        this.args = args;
        this.fun = fun;
    }
}

export = Block
