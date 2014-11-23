/// <reference path="../ref.ts" />

import numerics_ = require("./type/numerics");
import memory_ = require("./mem/memory");
import int8_ = require("./type/int8");
import uint8_ = require("./type/uint8");
import int16_ = require("./type/int16");
import uint16_ = require("./type/uint16");
import int32_ = require("./type/int32");
import uint32_ = require("./type/uint32");
import int64_ = require("./type/int64");

export var VERSION : number = 0.1;
export var numerics = numerics_;
export var int8 = int8_;
export var uint8 = uint8_;
export var int16 = int16_;
export var uint16 = uint16_;
export var int32 = int32_;
export var uint32 = uint32_;
export var int64 = int64_;
export var memory = memory_;
