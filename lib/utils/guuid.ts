
/// <reference path="../ref.ts" />

import rand = require("./rand");

function randomSymbol(space : string) : string {
	var n : number = rand.Int(space.length);
	return space[n];
}

function uuid() : string {
	var res : string = "";

	for (var i = 0; i < 36; i++) {
		if (i === 8 || i === 13 || i === 18 || i === 23) {
			res += "-";
		} else if (i == 14) {
			res += "4";
		} else if (i == 19) {
			res += randomSymbol("89AB");
		} else {
			res += randomSymbol("ABCDEF0123456789");
		}
	}
	return res;
}
