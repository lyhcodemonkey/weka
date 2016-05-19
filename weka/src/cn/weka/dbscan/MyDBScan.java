package cn.weka.dbscan;

import java.io.File;
import java.io.IOException;
import weka.clusterers.*;
import weka.core.*;
import weka.core.converters.*;

public class MyDBScan {
	public static void main(String[] args) throws Exception {
       // TODO Auto-generated method stub
       Instances ins=null;
       File file=new File("G:/�������O--Dģ��/ģ�ͽ�ģ�ṩ����1101/data2.arff");
       ArffLoader loader=new ArffLoader();
       try {
           DBSCAN dbs=new DBSCAN();
           loader.setFile(file);
           ins=loader.getDataSet();
           dbs.setEpsilon(1);//�����С
           dbs.setMinPoints(4);//��������С����
          dbs.buildClusterer(ins);
           System.out.println(dbs.toString());
           ClusterEvaluation eval = new ClusterEvaluation();
           eval.setClusterer(dbs);
           eval.evaluateClusterer(ins);
           System.out.println(ins.toString());
           double[] num = eval.getClusterAssignments();
              for (int i = 0; i < num.length; i++)
                {
                    System.out.println(String.valueOf( num[i]));
                }
              System.out.println(eval.clusterResultsToString());
              System.out.println(eval.getNumClusters());
       } catch (IOException e) {
           // TODO Auto-generated catch block
           e.printStackTrace();
       }
}
}