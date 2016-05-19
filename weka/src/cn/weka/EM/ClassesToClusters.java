package cn.weka.EM;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * EM��һ�ֻ���ģ�͵ľ����㷨�������������ϸ�˹���ģ��,�㷨��Ŀ����ȷ��������˹����֮��Ĳ����������ϸ������ݣ�
 * ���õ�һ��ģ�����࣬��ÿ�������Բ�ͬ��������ÿ����˹�ֲ���������ֵ�������ϸ�������á�
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