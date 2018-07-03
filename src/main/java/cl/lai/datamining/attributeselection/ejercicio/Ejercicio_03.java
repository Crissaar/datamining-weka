package cl.lai.datamining.attributeselection.ejercicio;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Discretize;


public class Ejercicio_03 
{
	public static Instances load_normal_data_set(String file)throws Exception{
		//Lectura de archivo general
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
		    	//Lectura de archivo para obtener los valores nominales
		    	 
		    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
		    	
		    	String aux = reader.readLine();
		    	int totalRegistros = 6329;
		    	
		    	String arrays[] = aux.split(";");
		    	
		    	Instances dptos = null;
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
		    	isTrainingSet.setClassIndex(classAttribute.index());
		     	int filas = 1;
		    	//Lectura de cada instancia
		      	while((aux=reader.readLine())!=null){
		     		arrays = aux.split(";");
		     		correspondencia.put(filas,arrays[0]);
		     		
		     		DenseInstance inst = new DenseInstance(attributes.size());
		     		for(int at=0,   i=2; i<arrays.length-1;i++,at++){
		     			double valor = Double.parseDouble(arrays[i]);
		      			inst.setValue(attributes.get(at), valor);
		        	}
		     		inst.setValue(classAttribute, arrays[arrays.length-1]);
		     		 
		     		isTrainingSet.add(inst);
		     		filas++;
		    		 
		    	} 
		      	return isTrainingSet;
	}
	
	public static void main( String[] args ) throws Exception{
		
		String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		String file = current + "arriendo_dpto_categoria_numerica_clasificacion.csv";
		
		Instances isTrainingSet = load_normal_data_set(file);
		
      	
      	
      	//TODO implementar lo que el documento señala
      	
	  	System.out.println("Atributos seleccionados");
	  	Enumeration<Attribute> enu = isTrainingSet.enumerateAttributes();
		while (enu.hasMoreElements()) {
			Attribute attr = enu.nextElement();
			System.out.println(attr);
		}
	  	
        
    }
}
