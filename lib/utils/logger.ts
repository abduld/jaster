﻿

enum LogType {
    Debug = 0,
    Trace = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4
}

class Logger {
    private _level: LogType;
    constructor(level?: LogType) {
        if (level) {
            this._level = level;
        } else {
            this._level = LogType.Debug;
        }
    }
    private go(msg: string, type: LogType) {
        var color: { [id: string]: string; } = {
            "LogType.Debug": '\033[39m',
            "LogType.Trace": '\033[39m',
            "LogType.Warn": '\033[33m',
            "LogType.Error": '\033[33m',
            "LogType.Fatal": '\033[31m'
        };
        if (type >= this._level) {
            console[type](color[type.toString()] + msg + color["LogType.Debug"]);
        }
    }
    debug(msg) { this.go(msg, LogType.Debug); }
    trace(msg: string) { this.go(msg, LogType.Trace); }
    warn(msg: string) { this.go(msg, LogType.Warn); }
    error(msg: string) { this.go(msg, LogType.Error); }
    fatal(msg: string) { this.go(msg, LogType.Fatal); }
};

export = Logger;