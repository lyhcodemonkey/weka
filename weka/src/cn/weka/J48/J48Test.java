package cn.weka.J48;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * J48 �������� C4.5 �㷨 C4.5 �㷨һ�ַ���������㷨 , ������㷨�� ID3 �㷨��C4.5 �㷨�̳��� ID3
 * �㷨���ŵ㣬�������¼������ ID3 �㷨�����˸Ľ��� 1������Ϣ��������ѡ�����ԣ��˷�������Ϣ����ѡ������ʱƫ��ѡ��ȡֵ������ԵĲ��㣻
 * 2��������������н��м�֦�� 3���ܹ���ɶ��������Ե���ɢ������ 4���ܹ��Բ��������ݽ��д��� C4.5 �㷨�������ŵ㣺
 * �����ķ������������⣬׼ȷ�ʽϸߡ� ��ȱ���ǣ��ڹ������Ĺ����У���Ҫ�����ݼ����ж�ε�˳��ɨ���������������㷨�ĵ�Ч��
 * http://www.ibm.com/developerworks/cn/opensource/os-cn-datamining/
 */
public class J48Test {

	public static void main(String[] args) throws Exception {
		Classifier m_classifier = new J48();
		// ѵ�������ļ����ٷ��Դ��� demo ����
		File inputFile = new File("E:/Program Files (x86)/Weka-3-6/data/cpu.with.vendor.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�
		// ���������ļ������ copy һ��ѵ���ļ��������������Ԥ��׼ȷ��У��
		inputFile = new File("E:/Program Files (x86)/Weka-3-6/data/cpu.with.vendor.arff");
		atf.setFile(inputFile);
		Instances instancesTest = atf.getDataSet(); // ��������ļ�
		instancesTest.setClassIndex(0); // ���÷������������кţ���һ��Ϊ0�ţ���instancesTest.numAttributes()����ȡ����������
		double sum = instancesTest.numInstances(), // ��������ʵ����
				right = 0.0f;
		instancesTrain.setClassIndex(0);// �������ԣ���һ���ֶ�
		m_classifier.buildClassifier(instancesTrain); // ѵ��
		for (int i = 0; i < sum; i++)// ���Է�����
		{
			double predicted = m_classifier.classifyInstance(instancesTest.instance(i));
			System.out.println(
					"Ԥ��ĳ����¼�ķ���id��" + predicted + ", ����ֵ��" + instancesTest.classAttribute().value((int) predicted));
			System.out.println(
					"�����ļ��ķ���ֵ�� " + instancesTest.instance(i).classValue() + ", ��¼��" + instancesTest.instance(i));
			System.out.println("--------------------------------------------------------------");

			// ���Ԥ��ֵ�ʹ�ֵ��ȣ����������еķ������ṩ����Ϊ��ȷ�𰸣�����������壩
			if (m_classifier.classifyInstance(instancesTest.instance(i)) == instancesTest.instance(i).classValue()) {
				right++;// ��ȷֵ��1
			}
		}
		// �뽫�ļ����ݵĵ�һ�� ? ������ȷ�𰸣��������з���Ԥ��Ľ����������ֻ�ǵ�����Ԥ�⣬��������û�����塣
		System.out.println("J48 classification precision:" + (right / sum));
	}

}
