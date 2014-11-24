
/// <reference path="../ref.ts" />


enum LogType {
	Debug = 0,
	Trace = 1,
	Warn = 2,
	Error = 3,
	Fatal = 4
}

class Logger {
	private level : LogType;
	constructor(level? : LogType) {
		if (level) {
			this.level = level;
		} else {
			this.level = LogType.Debug;
		}
	}
	private go(msg : string, type : logType) {
		var color = {
	      LogType.Debug: '\033[39m',
	      LogType.Trace: '\033[39m',
	      LogType.Warn: '\033[33m',
	      LogType.Error: '\033[33m',
	      LogType.Fatal: '\033[31m'
	    };
	    if (type >= this.level) {
	    	console[type](color[type] + msg + color[LogType.Debug]);
	    }
	}
	function debug(msg) {go(msg, LogType.Debug);}
	function trace(msg) {go(msg, LogType.Trace);}
	function warn(msg) {go(msg, LogType.Warn);}
	function error(msg) {go(msg, LogType.Error);}
	function fatal(msg) {go(msg, LogType.Fatal);}
}

export = Logger

