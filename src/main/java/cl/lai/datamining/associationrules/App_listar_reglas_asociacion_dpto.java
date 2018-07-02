package cl.lai.datamining.associationrules;

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
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class App_listar_reglas_asociacion_dpto 
{
	public static String file = "/Users/chunhaulai/Documents/workspace/lai-datamining-app/src/resources/arriendo_dpto_categoria.csv";
	public static void main( String[] args ) throws IOException{
		//Lectura de archivo general
    	BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	BufferedReader readerSoloGrupos = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
   	 
    	String aux = null;
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = null;
    	
     	List<HashSet<String>> my_nominal_grupos = new ArrayList<HashSet<String>>( ); 
    	int filas = 0;
    	while((aux=readerSoloGrupos.readLine())!=null){
    		String grupos[] = aux.split(";");
    		if(filas==0){
    			int columna =0;
     			for(String g: grupos){
     				//Por cada columna, se le crea un HashSet para almacenar los valores nominales
     				my_nominal_grupos.add(new HashSet<String>());
     				columna++;
     			}
    				
     		}else{
    			int columna =0;
     			for(String g: grupos){
     				//Por cada columna, se le insertan los valores nominales en el hash set correspondiente
     				my_nominal_grupos.get(columna).add(g);
    				columna++;
    			}
    		}
    		filas++;
    	}
    	filas = 0;
    	//Lectura de cada instancia
      	 while((aux=reader.readLine())!=null){
     		String grupos[] = aux.split(";");
    		if(filas==0){
    			//En la primera fila, se almacena los titulos de cada atributo (a excepción de la primera columna que contiene solamente id de la propiedad)
    			attributes = new ArrayList<Attribute>(grupos.length-1);
        		for(int i=1;i<grupos.length;i++){
        			Attribute attr = new Attribute(grupos[i],new ArrayList<String>(my_nominal_grupos.get(i)));
        			attributes.add(attr);
        		}
        		//Crear el objeto instancia con todos los atributos definidos
        		dptos = new Instances("dptos", attributes, 0);
    		}else{
    			//Definir la instancia de datos 
    			DenseInstance inst = new DenseInstance(attributes.size());
    			 
    			for(int i=1;i<grupos.length;i++){
    				//Omitir los valores que contiene >400
    				if( (">400".equalsIgnoreCase(grupos[i]) )){
    					inst.setMissing(attributes.get(i-1 ));
    				}else
    					inst.setValue(attributes.get(i-1 ),grupos[i]);
        		}
    			dptos.add(inst);
    		}
    		filas++;
    		 
    	}
      	
     	Apriori aprioriObj = new Apriori();
    	try {
    		//minimo soporte 0.1, minima confianza 0.5, cantidad de reglas: 30
    		String []options =  {"-C","0.5","-N","30","-M","0.1"};
    		aprioriObj.setOptions(options);
    		aprioriObj.buildAssociations(dptos);
    	} catch (Exception e) {
    		e.printStackTrace();
    	}
    	System.out.println(aprioriObj);
    }
}
