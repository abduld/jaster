
/// <reference path="../../ref.ts" />


import cuda = require("./cuda");

export class Thread {
    public threadIdx : utils.Dim3 = new utils.Dim3(0);
    public blockIdx : utils.Dim3 = new utils.Dim3(0);
    public blockDim : utils.Dim3 = new utils.Dim3(0);
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public block : cuda.BlockGroup;
    public threads : Thread[] = null;
}