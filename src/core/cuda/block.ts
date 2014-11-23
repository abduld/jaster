
/// <reference path="../../ref.ts" />
/// <reference path="cuda.ts" />

import cuda = require("./cuda");
import utils = require("./../../utils/utils");

class Block {
    public blockIdx : utils.Dim3 = new utils.Dim3(0);
    public blockDim : utils.Dim3 = new utils.Dim3(0);
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public threads : cuda.Thread[] = null;
}

export = Block
