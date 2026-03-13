#pragma once

#include "cnpy.h"
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

void unpackFeatures(cnpy::NpyArray x, cnpy::NpyArray& x_out,
                    cnpy::NpyArray& mask_out);

void oc_inference(const std::vector<std::vector<float>>& x,
                  const std::vector<float>& beta, std::vector<int>& object_ids,
                  double beta_thres, double dist_thres, int bkg_idx);

double rand_index(const std::vector<int>& truth, const std::vector<int>& pred);