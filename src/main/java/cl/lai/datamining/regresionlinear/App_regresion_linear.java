package cl.lai.datamining.regresionlinear;

 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import weka.classifiers.lazy.LWL;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class App_regresion_linear {
	public static String file = "/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria_numerica_5atributos.csv";

	public static void main(String[] args) throws Exception {
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 52;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	//tomar el primer atributo como target
    	Attribute classAttribute = new Attribute("precio");
    	attributes.add(classAttribute);
    	//considerar los 4 atributos restantes: 
    	for(int i=2; i<arrays.length;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClass(classAttribute);
    	
    	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		double predictiveSet[] = new double [arrays.length-1];
     		for(int i=1; i<arrays.length;i++){
     			double valor = Double.parseDouble(arrays[i]);
     			predictiveSet [i-1] = valor;
     		}
     		Instance inst = new DenseInstance(1,predictiveSet);
     		isTrainingSet.add(inst);
     		filas++;
    		 
    	}
			
			
		LWL l = new LWL();
		l.setKNN(3);
		l.buildClassifier(isTrainingSet);
		
		//double []valores_probar = new double[]{0,261,168,303,666};
		//DenseInstance inst = new DenseInstance(1,valores_probar);
		DenseInstance inst = new DenseInstance(5);
		
		inst.setValue(classAttribute, 0);
		Attribute a1 = attributes.get(1);//ALMACENES PEQUENOS VENTA DE ALIMENTOS
		Attribute a2 = attributes.get(2);//ESTABLECIMIENTOS DE ENSENANZA PRIMARIA Y SECUNDARIA PARA ADULTOS
		Attribute a3 = attributes.get(3);//MANTENIMIENTO Y REPARACION DE VEHICULOS AUTOMOTORES
		Attribute a4 = attributes.get(4);//PELUQUERIAS Y SALONES DE BELLEZA
		inst.setValue(a1, 1377);
		inst.setValue(a2,493);
		inst.setValue(a3,404);
		inst.setValue(a4,963);
		 
		//asociar la instancia a un dataset
		Instances dataset = new Instances("probar", attributes, 1);
		dataset.setClassIndex(0);
		inst.setDataset(dataset); 
		
		
		double precio = l.classifyInstance(inst);
		 
		System.out.println("Precio estimado: "+ precio);
  
	}
	
	 
}