module lib.wb {

    interface TimeEntry {
        category: string;
        message: string;
        startTime: number;
        endTime?: number;
        stopped: boolean;
    }
    var times_: Array<TimeEntry> = [];

    export function wbTime_start(state, category: string, ...args: any[]) {
        times_.unshift({
            category: category,
            message: args.join(""),
            stopped: false,
            startTime: lib.utils.timer.now()
        });
    }

    export function wbTime_stop(state, category: string, ...args: any[]) {
        var timer = _.find(times_, { 'category': category, 'stopped': false });
        if (lib.utils.isUndefined(timer)) {
            return;
        }
        times_.unshift({
            category: category,
            message: args.join(""),
            stopped: true,
            startTime: timer.startTime,
            endTime: lib.utils.timer.now()
        });
    }
}
