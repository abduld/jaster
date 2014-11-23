/// <reference path="../ref.ts" />

import numerics_ = require("./type/numerics");
import memory_ = require("./mem/memory");


export var VERSION : number = 0.1;
export var numerics = numerics_;
export var int8 = numerics_.int8;
export var uint8 = numerics.uint8;
export var int16 = numerics.int16;
export var uint16 = numerics.uint16;
export var int32 = numerics.int32;
export var uint32 = numerics.uint32;
export var int64 = numerics.int64;
export var memory = memory_;
