
/// <reference path="utils.ts" />

module Core {
export class Thread {
        private id : string;

private threadIdx : Dim3;
private blockIdx : Dim3;
private blockDim : Dim3;
private gridDim : Dim3;
private gridIdx : Dim3;
    constructor(id : string) {
        this.id = id;
    }
    }
}