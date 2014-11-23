
/// <reference path="../../ref.ts" />


import cuda = require("./cuda");
import block = require("./block");
import thread = require("./thread");
import grid = require("./grid");
import warp = require("./warp");
import utils = require("./../../utils/utils");

class Thread {
	error : utils.Error = new utils.Error();
    threadIdx : utils.Dim3 = new utils.Dim3(0);
    blockIdx : utils.Dim3 = new utils.Dim3(0);
    blockDim : utils.Dim3 = new utils.Dim3(0);
    gridIdx : utils.Dim3 = new utils.Dim3(0);
    gridDim : utils.Dim3 = new utils.Dim3(0);
    block : block;
    warp : warp;
    status : cuda.Status;
    fun : Function = undefined;
    args : Array<any> = [];
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
    public run() {
        this.status = cuda.Status.Running;
        var res = this.fun.apply(this, this.args);
        this.status = cuda.Status.Complete;
        return res;
    }

}

export = Thread