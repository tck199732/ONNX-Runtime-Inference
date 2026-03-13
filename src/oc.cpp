#include "oc.hpp"

void unpackFeatures(cnpy::NpyArray x, cnpy::NpyArray& x_out,
                    cnpy::NpyArray& mask_out)
{
    auto shape = x.shape;
    auto ndims = x.shape.size();
    assert(static_cast<int>(ndims) == 2);

    auto numNodes = shape[0];
    auto numFeatures = shape[1];
    auto numPairs = numFeatures / 2;

    x_out = cnpy::NpyArray({numNodes, numPairs, 2}, sizeof(float), false);
    mask_out = cnpy::NpyArray({numNodes, numPairs}, sizeof(bool), false);

    for (int iNode = 0; iNode < numNodes; ++iNode)
    {
        for (int iPair = 0; iPair < numPairs; ++iPair)
        {
            float energy = x.data<float>()[iNode * numFeatures + 2 * iPair];
            float time = x.data<float>()[iNode * numFeatures + 2 * iPair + 1];
            x_out.data<float>()[iNode * numPairs * 2 + iPair * 2] = energy;
            x_out.data<float>()[iNode * numPairs * 2 + iPair * 2 + 1] = time;
            mask_out.data<bool>()[iNode * numPairs + iPair] =
                (std::abs(energy) > 0.0f) || (std::abs(time) > 0.0f);
        }
    }
    return;
}

void oc_inference(const std::vector<std::vector<float>>& x,
                  const std::vector<float>& beta, std::vector<int>& object_ids,
                  double beta_thres, double dist_thres, int bkg_idx)
{

    object_ids.clear();
    object_ids.resize(x.size());
    std::vector<unsigned int> seed_indices;
    for (int i = 0; i < x.size(); ++i)
    {
        if (beta[i] > beta_thres)
        {
            seed_indices.push_back(i);
        }
    }

    if (seed_indices.empty())
    {
        for (int i = 0; i < x.size(); ++i)
        {
            object_ids[i] = bkg_idx;
        }
        return;
    }

    std::vector<std::vector<float>> dists(
        x.size(), std::vector<float>(seed_indices.size()));

    for (int i = 0; i < x.size(); ++i)
    {
        for (int s = 0; s < seed_indices.size(); ++s)
        {
            dists[i][s] = 0.0;
            for (int j = 0; j < x[0].size(); ++j)
            {
                dists[i][s] += std::pow(x[i][j] - x[seed_indices[s]][j], 2.0);
            }
            dists[i][s] = std::sqrt(dists[i][s]);
        }
    }

    for (int i = 0; i < x.size(); ++i)
    {
        float min_d = 1e4;
        int idx;
        for (int j = 0; j < seed_indices.size(); j++)
        {
            if (dists[i][j] < min_d)
            {
                min_d = dists[i][j];
                idx = j;
            }
        }
        object_ids[i] = (min_d > dist_thres) ? bkg_idx : seed_indices[idx];
    }
}

double rand_index(const std::vector<int>& truth, const std::vector<int>& pred)
{
    int n = truth.size();
    int agree = 0;
    int total = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            bool same_truth = (truth[i] == truth[j]);
            bool same_pred = (pred[i] == pred[j]);

            if (same_truth == same_pred)
            {
                agree++;
            }

            total++;
        }
    }

    return (double)agree / total;
}