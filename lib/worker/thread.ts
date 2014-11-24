
/// <reference path="../../ref.ts" />

import numerics = require("../type/numerics");
import integer = require("../type/integer");
import utils = require("../../utils/utils");
export class Thread {
        private id : string;

private threadIdx : utils.Dim3;
private blockIdx : utils.Dim3;
private blockDim : utils.Dim3;
private gridDim : utils.Dim3;
private gridIdx : utils.Dim3;
    constructor(id : string) {
        this.id = id;
    }
    }