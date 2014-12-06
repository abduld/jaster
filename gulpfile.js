var gulp = require('gulp');
var gutil = require('gulp-util');
var source = require('vinyl-source-stream');
var watchify = require('watchify');
var browserify = require('browserify');

gulp.task('browserify', function () {
    var browserify = require('browserify');
    var source = require('vinyl-source-stream');
    var mold = require('mold-source-map');
    var buffer = require('gulp-buffer');
    var gulpif = require('gulp-if');
    var size = require('gulp-size');
    var notify = require('gulp-notify');
    var fixWindowsSourceMaps = require('gulp-fix-windows-source-maps');
    var isWin = Boolean(~process.platform.indexOf('win'));
    var appRoot = '.';
    var handleErrors = function() {

        var args = Array.prototype.slice.call(arguments);

        // Send error to notification center with gulp-notify
        notify.onError({
            title: "Compile Error",
            message: "<%= error.message %>"
        }).apply(this, args);

        // Keep gulp from hanging on this task
        this.emit('end');
    };


    return browserify({
        entries: appRoot + '/app.ts',
    debug: true
    })
        .plugin('tsify', {target: 'ES5', module: 'commonjs'})
        .on('error', handleErrors)

        .bundle()
        .on('error', handleErrors)

        // vinyl-source-stream makes the bundle compatible with gulp
        .pipe(mold.transformSourcesRelativeTo(appRoot))
        .pipe(source('bundle.js')) // Desired filename
        .pipe(buffer())
        .pipe(gulpif(isWin, fixWindowsSourceMaps()))
        // Output the file
        .pipe(gulp.dest('./app/'))
        .pipe(size({title: 'browserify bundle size'}));
});

gulp.task('watch', function() {
    watchify.args.debug = true;
    var browserify = require('browserify');
    var source = require('vinyl-source-stream');
    var mold = require('mold-source-map');
    var buffer = require('gulp-buffer');
    var gulpif = require('gulp-if');
    var size = require('gulp-size');
    var notify = require('gulp-notify');
    var fixWindowsSourceMaps = require('gulp-fix-windows-source-maps');
    var isWin = Boolean(~process.platform.indexOf('win'));
    var appRoot = '.';
    var handleErrors = function() {

        var args = Array.prototype.slice.call(arguments);

        // Send error to notification center with gulp-notify
        notify.onError({
            title: "Compile Error",
            message: "<%= error.message %>"
        }).apply(this, args);

        // Keep gulp from hanging on this task
        this.emit('end');
    };


    var bundler =  browserify({
        entries: appRoot + '/app.ts',
        debug: true
    })
        .plugin('tsify', {target: 'ES5', module: 'commonjs'})
        .on('error', handleErrors);
    var watcher  = watchify(bundler);

    return watcher
        .on('update', function () { // When any files update
            var updateStart = Date.now();
            console.log('Updating!');
            watcher.bundle() // Create new bundle that uses the cache for high performance
                .pipe(source('app.js'))
                // This is where you add uglifying etc.
                .pipe(gulp.dest('./build/'));
            console.log('Updated!', (Date.now() - updateStart) + 'ms');
        })
        .bundle() // Create the initial bundle when starting the task

        .on('error', handleErrors)

        // vinyl-source-stream makes the bundle compatible with gulp
        .pipe(mold.transformSourcesRelativeTo(appRoot))
        .pipe(source('bundle.js')) // Desired filename
        .pipe(buffer())
        .pipe(gulpif(isWin, fixWindowsSourceMaps()))
        // Output the file
        .pipe(gulp.dest('./app/'))
        .pipe(size({title: 'browserify bundle size'}));
});