package cl.lai.datamining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.function.BiConsumer;

import javax.smartcardio.ATR;

import weka.associations.Apriori;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class App_transformacion_numerica 
{
	public static String file = "/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria.csv";
	
	public static <E> void forEach(
	        Iterable<? extends E> elements, BiConsumer<Integer, ? super E> action) {
	        Objects.requireNonNull(elements);
	        Objects.requireNonNull(action);
	
	        int index = 0;
	        for (E element : elements) {
	            action.accept(index++, element);
	        }
    	}

 
	public static void main( String[] args ) throws IOException{
		//Lectura de archivo general
    	BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	PrintWriter writer = new PrintWriter(  new File("/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria_numerica.csv"));

    	String aux = reader.readLine();
    	
    	writer.println(aux+";");
    	
    	while((aux=reader.readLine())!=null){
    		StringBuffer buffer = new StringBuffer();
    		forEach(Arrays.asList(aux.split(";")), (index, str) -> {
    			
    			if(index > 1){
    				switch(str){
    					case "0-100":buffer.append((int)(Math.random()*100 + 1)).append(";");break;
    					case "101-200":buffer.append((int)(Math.random()*100 + 101)).append(";");break;
    					case "201-300":buffer.append((int)(Math.random()*100 + 201)).append(";");break;
    					case "301-400":buffer.append((int)(Math.random()*100 + 301)).append(";");break;
    					case ">400":buffer.append((int)(Math.random()*1000 + 401)).append(";");break;
    				} 
    			}else{
    				buffer.append(str).append(";");
    			}
    		});
     		writer.println(buffer.toString());
     		writer.flush();
     	}
    	writer.println();
    }
}
