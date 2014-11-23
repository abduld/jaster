
/// <reference path="../../ref.ts" />

import cuda = require("./cuda");
import utils = require("./../../utils/utils");

class Grid {
    public gridIdx : utils.Dim3 = new utils.Dim3(0);
    public gridDim : utils.Dim3 = new utils.Dim3(0);
    public blocks : cuda.Block[] = null;
}

export = Grid