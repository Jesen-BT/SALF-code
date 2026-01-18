from skmultiflow.data.file_stream import FileStream



class incre_stream:


    def __init__(self, filepath, change_times, feature_num=None, fixed_features=None):
        self.stream = FileStream(filepath)
        self.n_samples = self.stream.n_samples
        self.target_values = self.stream.target_values

        self.change_times = int(change_times)
        self.feature_num = feature_num

        self.original_n_features = int(self.stream.n_features)
        self.change_samples = max(1, self.n_samples // (self.change_times + 1))
        self.features_per_change = max(1, self.original_n_features // (self.change_times + 1))


        self.active_features = max(1, self.original_n_features // (self.change_times + 1))


        if fixed_features is None:
            self.fixed_features = int(self.active_features)
        else:
            self.fixed_features = int(fixed_features)
        self.fixed_features = max(1, min(self.fixed_features, self.original_n_features))


        self.current_sample = 0
        self.current_change = 0
        self.next_change_point = self.change_samples

    def has_more_samples(self):
        return self.stream.has_more_samples()

    def restart(self):
        self.stream.restart()
        self.current_sample = 0
        self.current_change = 0
        self.active_features = max(1, self.original_n_features // (self.change_times + 1))

        self.next_change_point = self.change_samples

    def get_current_active_features(self):

        return int(self.active_features)

    def get_fixed_features(self):

        return int(self.fixed_features)

    def next_sample(self, batch_size=1, return_fixed=False, return_all=False):

        X, y = self.stream.next_sample(batch_size)
        self.current_sample += int(batch_size)


        while (self.current_change < self.change_times and
               self.current_sample >= self.next_change_point):
            self._add_features()
            self.current_change += 1
            self.next_change_point += self.change_samples

        X_inc = X[:, :self.active_features]

        if not return_fixed and not return_all:
            return X_inc, y
        elif return_fixed and not return_all:
            X_fixed = X[:, :self.fixed_features]
            return X_inc, X_fixed, y
        elif not return_fixed and return_all:
            return X_inc, X, y
        elif return_fixed and return_all:
            X_fixed = X[:, :self.fixed_features]
            return X_inc, X_fixed, X, y

    def _add_features(self):

        if self.feature_num is None:
            self.active_features += self.features_per_change
        else:
            self.active_features += int(self.feature_num)

        if self.active_features > self.original_n_features:
            self.active_features = self.original_n_features
