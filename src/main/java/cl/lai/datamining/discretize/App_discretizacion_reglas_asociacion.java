package cl.lai.datamining.discretize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import weka.associations.Apriori;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class App_discretizacion_reglas_asociacion 
{
	public static String file = "/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria_numerica_5atributos_clasificacion.csv";
	public static void main( String[] args ) throws Exception{
		//Lectura de archivo general
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 52;
    	
    	String arrays[] = aux.split(";");
    	
     	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	//omitir el primer, segundo atributo y el último atributo, y considerar los 4 atributos restantes: 
    	for(int i=2; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	//considerar el ultimo atributo como target 
    	ArrayList<String> clasesPreviamenteDefinida = new ArrayList<String>(3); 
    	clasesPreviamenteDefinida.add("BAJO"); 
    	clasesPreviamenteDefinida.add("INTERMEDIO"); 
    	clasesPreviamenteDefinida.add("ALTO");
    	Attribute classAttribute = new Attribute("SECTOR",clasesPreviamenteDefinida);
    	attributes.add(classAttribute);
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	//isTrainingSet.setClassIndex(classAttribute.index());
    	
    	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(5);
     		for(int at=0,   i=2; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
     			inst.setValue(attributes.get(at), valor);
        	}
     		inst.setValue(classAttribute, arrays[arrays.length-1]);
     		 
     		isTrainingSet.add(inst);
     		filas++;
    		 
    	} 
      	
      	Discretize discretizeNumeric = new Discretize();
      	discretizeNumeric.setInputFormat(isTrainingSet); 
    	discretizeNumeric.setOptions(new String[] {
    			"-B",  "3",  // numero de division
    			"-R",  "first-last"}); //rango de numero (desde primer atributo hasta ultimo atributo )
    	isTrainingSet = Filter.useFilter(isTrainingSet, discretizeNumeric);        

    	Apriori aprioriObj = new Apriori();
    	try {
    		//minimo soporte 0.5, minima confianza 0.9, cantidad de reglas: 10
    		String []options =  {"-C","0.95","-N","10","-M","0.5"};
    		aprioriObj.setOptions(options);
    		aprioriObj.buildAssociations(isTrainingSet);
    	} catch (Exception e) {
    		e.printStackTrace();
    	}
    	System.out.println(aprioriObj);
    }
}
