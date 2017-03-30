/**
 * @file emst_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Calls the DualTreeBoruvka algorithm from dtb.hpp.
 * Can optionally call naive Boruvka's method.
 *
 * For algorithm details, see:
 *
 * @code
 * @inproceedings{
 *   author = {March, W.B., Ram, P., and Gray, A.G.},
 *   title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *      Applications.}},
 *   booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
 *      on Knowledge Discovery and Data Mining}
 *   series = {KDD 2010},
 *   year = {2010}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "dtb.hpp"

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>

PROGRAM_INFO("Fast Euclidean Minimum Spanning Tree", "This program can compute "
    "the Euclidean minimum spanning tree of a set of input points using the "
    "dual-tree Boruvka algorithm."
    "\n\n"
    "The output is saved in a three-column matrix, where each row indicates an "
    "edge.  The first column corresponds to the lesser index of the edge; the "
    "second column corresponds to the greater index of the edge; and the third "
    "column corresponds to the distance between the two points.");

PARAM_MATRIX_IN_REQ("input", "Input data matrix.", "i");
PARAM_MATRIX_OUT("output", "Output data.  Stored as an edge list.", "o");
PARAM_FLAG("naive", "Compute the MST using O(n^2) naive algorithm.", "n");
PARAM_INT_IN("leaf_size", "Leaf size in the kd-tree.  One-element leaves give "
    "the empirically best performance, but at the cost of greater memory "
    "requirements.", "l", 1);

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace std;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file is not specified, so no output will be saved!"
        << endl;

  arma::mat dataPoints = std::move(CLI::GetParam<arma::mat>("input"));

  Log::Info << "Building tree.\n";

  // Check that the leaf size is reasonable.
  if (CLI::GetParam<int>("leaf_size") <= 0)
  {
    Log::Fatal << "Invalid leaf size (" << CLI::GetParam<int>("leaf_size")
        << ")!  Must be greater than or equal to 1." << std::endl;
  }

  // Initialize the tree and get ready to compute the MST.  Compute the tree
  // by hand.
  const size_t leafSize = (size_t) CLI::GetParam<int>("leaf_size");

  Timer::Start("tree_building");
  std::vector<size_t> oldFromNew;
  KDTree<EuclideanDistance, DTBStat, arma::mat> tree(dataPoints, oldFromNew,
      leafSize);
  metric::LMetric<2, true> metric;
  Timer::Stop("tree_building");

  DualTreeBoruvka<> dtb(&tree, metric);

  // Run the DTB algorithm.
  Log::Info << "Calculating minimum spanning tree." << endl;
  arma::mat results;
  dtb.ComputeMST(results);

  
  arma::mat valid_edges(dataPoints.n_cols,1);
  for(size_t i = 0;i < results.n_cols; ++i) {
  	if(results(2,i) < 1.1) {
  		valid_edges(i,0) = 1;
  	}
  }

  // Unmap the results.
  arma::mat unmappedResults(results.n_rows, results.n_cols);
  arma::mat cluster(dataPoints.n_cols,1);
  size_t cluster_number = 0;
  cluster_number += 1;
  
  for(size_t k = 0; k < results.n_cols; ++k) {
  	  if(cluster(results(0,k),0) == 0 && cluster(results(1,k),0) == 0) {
	  	  cluster(results(0,k),0) = cluster_number;
	  	  cluster(results(1,k),0) = cluster_number;
		  for(size_t i = 0;i < results.n_cols; ++i) {
		    	bool no_change = true;
		    	if(cluster(i,0) == cluster_number) {
			      	for(size_t j = 0; j < results.n_cols; ++j) {
			        	if((cluster(results(1,j),0) == 0 && results(0,j) == i) && valid_edges(j,0) == 1) {
			          		no_change = false;
			          		cluster(results(1,j),0) = cluster_number;
			        	} else if((cluster(results(0,j),0) == 0 && results(1,j) == i) && valid_edges(j,0) == 1) {
			          		no_change = false;           
			          		cluster(results(0,j),0) = cluster_number;
			        	}  
			      	}
			      	if(!no_change) {
			        	i = 0; 
			      	}
		    	}
		  	}
		  	cluster_number+=1;
		} else {
			if(cluster(results(0,k),0) == 0) {
				cluster(results(0,k),0) = cluster(results(1,k),0);
			} else {
				cluster(results(1,k),0) = cluster(results(0,k),0);
			}
		}
 	}

  for(size_t i = 0; i < results.n_cols; ++i) {
    cout<<results(0,i)<<" "<<results(1,i)<<" "<<cluster(results(0,i),0)<<endl;
  }

  for (size_t i = 0; i < results.n_cols; ++i)
  {
    
    for(size_t j = 0; j < results.n_rows; ++j) {
        unmappedResults(j, i) = dataPoints(j,i);  
    }
    //unmappedResults(2, i) = results(2, i);
  }

  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(unmappedResults);
}
