/// <reference path="utils.ts" />
module Parallel {
	enum WorkerStatus {
		Paused,
		Idle,
		Busy,
		Cancel
	};
	export class Worker {
		private id : string;
		private status : WorkerStatus;
		private fun : (... args : any[]) => any;
		private pause_length : number = 100; // milliseconds
		constructor() {
			this.id = Core.guuid();
			this.status = WorkerStatus.Idle;
		}
		private run0(init : number, end : number, inc : number) : boolean {
			var iter : number = init;
			if (this.status === WorkerStatus.Paused) {
				return setTimeOut(run0, 2*pause_length, [init, end, inc]);
			}
			while (iter < end) {
				fun(iter);
				if (this.status === WorkerStatus.Cancel) {
					break ;
				} else if (this.status === WorkerStatus.Paused) {
					return setTimeOut(function() {run0, pause_length, [iter + inc, end, inc]);
				}
				iter += inc;
			}
			this.status = WorkerStatus.Idle;
		} 
		public run(start_idx : number, end_idx : number, inc? : number) : boolean {
			this.status = WorkerStatus = Busy;
			if (inc) {
				return run0(start_idx, end_idx, inc);
			} else {
				return run0(start_idx, end_idx, 1);
			}
		}
		public cancel() {
			this.status = 
		}
	}
}