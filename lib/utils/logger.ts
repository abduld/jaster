
module lib {
    export module utils {
        export module detail {

            enum LogType {
                Debug = 0,
                Trace = 1,
                Warn = 2,
                Error = 3,
                Fatal = 4
            }

            export class Logger {
                private _level: LogType;
                constructor(level?: LogType) {
                    if (level) {
                        this._level = level;
                    } else {
                        this._level = LogType.Debug;
                    }
                }
                private _go(msg: string, type: LogType) {
                    var color: { [id: string]: string; } = {
                        "LogType.Debug": '\033[39m',
                        "LogType.Trace": '\033[39m',
                        "LogType.Warn": '\033[33m',
                        "LogType.Error": '\033[33m',
                        "LogType.Fatal": '\033[31m'
                    };
                    if (type >= this._level) {
                        switch (type) {
                            case LogType.Debug:
                            case LogType.Trace:
                                console.info(msg);
                                break;
                            case LogType.Warn:
                                console.warn(msg);
                                break;
                            case LogType.Error:
                            case LogType.Fatal:
                            default:
                                debugger;
                                console.error(msg);
                                break;
                        }
                    }
                }
                debug(msg) { this._go(msg, LogType.Debug); }
                trace(msg: string) { this._go(msg, LogType.Trace); }
                warn(msg: string) { this._go(msg, LogType.Warn); }
                error(msg: string) { this._go(msg, LogType.Error); }
                fatal(msg: string) { this._go(msg, LogType.Fatal); }
            }
        }

        export var logger = new detail.Logger();
    }
}