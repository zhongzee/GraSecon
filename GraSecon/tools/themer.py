import torch


class Themer:
    def __init__(self, method, thresh=1, alpha=0.5):
        if method not in ['mean', 'peigen', 'mixed', 'all_eigens', 'weighted','mix_weighted']:
            raise NameError(f"{method} is not supported")

        self.method = method
        self.T = thresh
        self.alpha = alpha

    def _get_principal_eigenvector(self, stacked_feats):
        # Single child case
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0] / torch.norm(stacked_feats)
        print("========== used Peigen when shape={}".format(stacked_feats.shape))

        # Convert to float32 (clip feat is in float16)
        stacked_feats32 = stacked_feats.to(torch.float32)

        # Compute principal eigenvector
        U, S, V = torch.svd(stacked_feats32)            # SVD decomposition
        peigen_v = V[:, 0]                              # principal eigenvector
        peigen_v = peigen_v / torch.norm(peigen_v)      # L2-normalization
        peigen_v = peigen_v.to(torch.float16)           # convert it back to float16 for CLIP
        return peigen_v

    def _get_all_eigenvector(self, stacked_feats):
        # Single child case
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0] / torch.norm(stacked_feats)
        print("========== used ALL eigens when shape={}".format(stacked_feats.shape))

        # Convert to float32 (clip feat is in float16)
        stacked_feats32 = stacked_feats.to(torch.float32)

        # Compute principal eigenvector
        U, S, V = torch.svd(stacked_feats32)            # SVD decomposition
        normalized_weights_s = S / torch.sum(S)
        weighted_avg_v = torch.zeros_like(V[:, 0])

        for i in range(V.size(1)):
            weighted_avg_v += normalized_weights_s[i] * V[:, i]

        weighted_avg_v = weighted_avg_v / torch.norm(weighted_avg_v)
        weighted_avg_v = weighted_avg_v.to(torch.float16)
        return weighted_avg_v                       # weighted average

    def _get_mean_vector(self, stacked_feats):
        print("========== used Mean when shape={}".format(stacked_feats.shape))
        mean_theme = torch.mean(stacked_feats, dim=0)   # mean vector
        return mean_theme

    def _get_mixed_vector(self, stacked_feats):
        print("========== used MIXED when shape={}".format(stacked_feats.shape))
        mean_v = self._get_mean_vector(stacked_feats)
        peigen_v = self._get_principal_eigenvector(stacked_feats)
        mixed_v = self.alpha * mean_v + (1 - self.alpha) * peigen_v
        return mixed_v / torch.norm(mixed_v)

    def _get_weighted_vector(self, stacked_feats, weights):
        print("========== used WEIGHTED when shape={} with weights={}".format(stacked_feats.shape, weights))
        if len(weights) != stacked_feats.shape[0]:
            raise ValueError("Number of weights must match the number of features.")
        weighted_sum = sum(w * f for w, f in zip(weights, stacked_feats))
        weighted_theme = weighted_sum / sum(weights) # torch.Size([1024])
        return weighted_theme / torch.norm(weighted_theme)

    def _get_mix_weighted_vector(self, stacked_feats, weights):
        print("========== used MIX_WEIGHTED when shape={} with weights={}".format(stacked_feats.shape, weights))
        if len(weights) != stacked_feats.shape[0]:
            raise ValueError("Number of weights must match the number of features.")
        weights = torch.tensor(weights, dtype=stacked_feats.dtype, device=stacked_feats.device)
        # 计算加权和
        weighted_sum = torch.sum(weights.unsqueeze(1) * stacked_feats, dim=0)
        weighted_sum = weighted_sum / weights.sum()
        # 计算均值
        mean_feature = torch.mean(stacked_feats, dim=0)
        # 混合加权和与均值
        aggregated_feature = self.alpha * weighted_sum + (1 - self.alpha) * mean_feature
        return aggregated_feature

    def get_theme(self, stacked_feats, weights=None):
        if self.method == 'peigen' and stacked_feats.shape[0] > self.T:
            return self._get_principal_eigenvector(stacked_feats)
        elif self.method == 'all_eigens' and stacked_feats.shape[0] > self.T:
            return self._get_all_eigenvector(stacked_feats)
        elif self.method == 'mixed' and stacked_feats.shape[0] > self.T:
            return self._get_mixed_vector(stacked_feats)
        elif self.method == 'weighted' and weights is not None:
            return self._get_weighted_vector(stacked_feats, weights)
        elif self.method == 'mix_weighted' and weights is not None:
            return self._get_mix_weighted_vector(stacked_feats, weights)
        else:
            return self._get_mean_vector(stacked_feats)


if __name__ == '__main__':
    analyzer = Themer(method='weighted', thresh=1)
    stacked_feats = torch.rand((5, 10))  # Replace this with your actual data
    weights = [0.1, 0.3, 0.5, 0.8, 1.0]  # Example weights

    result = analyzer.get_theme(stacked_feats, weights=weights)
    print(result)
