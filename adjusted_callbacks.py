from keras import callbacks
import numpy as np
import warnings



class ModelCheckpoint(callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', previous_best=np.Inf, period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = previous_best
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)





class EarlyStopping(callbacks.Callback):
	"""Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
	"""

	def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
				 wait = 0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
		super(EarlyStopping, self).__init__()

		self.monitor = monitor
		self.baseline = baseline
		self.patience = patience
		self.verbose = verbose
		self.min_delta = min_delta
		self.wait = wait
		self.stopped_epoch = 0
		self.restore_best_weights = restore_best_weights
		self.best_weights = None

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
		elif mode == 'max':
			self.monitor_op = np.greater
		else:
			if 'acc' in self.monitor:
				self.monitor_op = np.greater
			else:
				self.monitor_op = np.less

		if self.monitor_op == np.greater:
			self.min_delta *= 1
		else:
			self.min_delta *= -1

	def on_train_begin(self, logs=None):
		# Allow instances to be re-used
		self.wait = 0
		self.stopped_epoch = 0
		if self.baseline is not None:
			self.best = self.baseline
		else:
			self.best = np.Inf if self.monitor_op == np.less else -np.Inf

	def on_epoch_end(self, epoch, logs=None):
		current = self.get_monitor_value(logs)
		if current is None:
			return

		if self.monitor_op(current - self.min_delta, self.best):
			self.best = current
			self.wait = 0
			if self.restore_best_weights:
				self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				if self.restore_best_weights:
					if self.verbose > 0:
						print('Restoring model weights from the end of '
							  'the best epoch')
					self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0 and self.verbose > 0:
			print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

	def get_monitor_value(self, logs):
		monitor_value = logs.get(self.monitor)
		if monitor_value is None:
			warnings.warn(
					'Early stopping conditioned on metric `%s` '
					'which is not available. Available metrics are: %s' %
					(self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
		return monitor_value













