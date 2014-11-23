
/// <reference path="../../ref.ts" />


import cuda = require("./cuda");
import utils = require("./../../utils/utils");

class Warp {
    public id : string = utils.guuid();
    public thread : cuda.Thead;
}

export = Warp