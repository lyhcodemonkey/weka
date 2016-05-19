package cn.weka.EM;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * EM是一种基于模型的聚类算法，假设样本符合高斯混合模型,算法的目的是确定各个高斯部件之间的参数，充分拟合给定数据，
 * 并得到一个模糊聚类，即每个样本以不同概率属于每个高斯分布，概率数值将由以上个参数获得。
 */
public class ClassesToClusters {
	public static void main(String[] args) throws Exception {
		// load data
		Instances data = DataSource.read("E:/Program Files (x86)/Weka-3-6/data/iris.arff");
		data.setClassIndex(data.numAttributes() - 1);

		// generate data for clusterer (w/o class)
		Remove filter = new Remove();
		filter.setAttributeIndices("" + (data.classIndex() + 1));
		filter.setInputFormat(data);
		Instances dataClusterer = Filter.useFilter(data, filter);

		// train clusterer
		EM clusterer = new EM();
		// set further options for EM, if necessary...
		clusterer.buildClusterer(dataClusterer);

		// evaluate clusterer
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(clusterer);
		eval.evaluateClusterer(data);

		// print results
		System.out.println(eval.clusterResultsToString());
	}
}