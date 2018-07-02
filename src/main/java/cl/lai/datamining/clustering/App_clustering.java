package cl.lai.datamining.clustering;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.smartcardio.ATR;

import weka.associations.Apriori;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class App_clustering 
{
	public static String file = "/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria_numerica.csv";
	public static void main( String[] args ) throws Exception{
		//Lectura de archivo general
    	BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 6329;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();;
    	for(int i=1; i<arrays.length;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
     	 
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	
    	int filas = 1;
    	//Lectura de cada instancia
      	 while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		double predictiveSet[] = new double [arrays.length-1];
     		for(int i=1; i<arrays.length;i++){
     			double distancia = Double.parseDouble(arrays[i]);
     			predictiveSet [i-1] = distancia;
     		}
     		Instance inst = new DenseInstance(1,predictiveSet);
     		isTrainingSet.add(inst);
     		filas++;
    		 
    	}
      	SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(20);
        kMeans.buildClusterer(isTrainingSet); 
        System.out.println(kMeans.getClusterCentroids());
        for (int i = 0; i < isTrainingSet.numInstances(); i++) { 
            System.out.println(correspondencia.get(i)+ " esta en cluster " + kMeans.clusterInstance(isTrainingSet.instance(i)) ); 
             
        } 
    }
}
