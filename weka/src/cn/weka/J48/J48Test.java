package cn.weka.J48;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * J48 即决策树 C4.5 算法 C4.5 算法一种分类决策树算法 , 其核心算法是 ID3 算法。C4.5 算法继承了 ID3
 * 算法的优点，并在以下几方面对 ID3 算法进行了改进： 1、用信息增益率来选择属性，克服了用信息增益选择属性时偏向选择取值多的属性的不足；
 * 2、在树构造过程中进行剪枝； 3、能够完成对连续属性的离散化处理； 4、能够对不完整数据进行处理。 C4.5 算法有如下优点：
 * 产生的分类规则易于理解，准确率较高。 其缺点是：在构造树的过程中，需要对数据集进行多次的顺序扫描和排序，因而导致算法的低效。
 * http://www.ibm.com/developerworks/cn/opensource/os-cn-datamining/
 */
public class J48Test {

	public static void main(String[] args) throws Exception {
		Classifier m_classifier = new J48();
		// 训练语料文件，官方自带的 demo 里有
		File inputFile = new File("E:/Program Files (x86)/Weka-3-6/data/cpu.with.vendor.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instancesTrain = atf.getDataSet(); // 读入训练文件
		// 测试语料文件：随便 copy 一段训练文件出来，做分类的预测准确性校验
		inputFile = new File("E:/Program Files (x86)/Weka-3-6/data/cpu.with.vendor.arff");
		atf.setFile(inputFile);
		Instances instancesTest = atf.getDataSet(); // 读入测试文件
		instancesTest.setClassIndex(0); // 设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数
		double sum = instancesTest.numInstances(), // 测试语料实例数
				right = 0.0f;
		instancesTrain.setClassIndex(0);// 分类属性：第一个字段
		m_classifier.buildClassifier(instancesTrain); // 训练
		for (int i = 0; i < sum; i++)// 测试分类结果
		{
			double predicted = m_classifier.classifyInstance(instancesTest.instance(i));
			System.out.println(
					"预测某条记录的分类id：" + predicted + ", 分类值：" + instancesTest.classAttribute().value((int) predicted));
			System.out.println(
					"测试文件的分类值： " + instancesTest.instance(i).classValue() + ", 记录：" + instancesTest.instance(i));
			System.out.println("--------------------------------------------------------------");

			// 如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）
			if (m_classifier.classifyInstance(instancesTest.instance(i)) == instancesTest.instance(i).classValue()) {
				right++;// 正确值加1
			}
		}
		// 请将文件内容的第一列 ? 换成正确答案，才能评判分类预测的结果，本例中只是单纯的预测，下面的输出没有意义。
		System.out.println("J48 classification precision:" + (right / sum));
	}

}
