import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {

    public static void main(String[] args) {
        Map<String, List<List<String>>> map = new HashMap<>();
        // dataset -> [ [...] (space separated elements) ] (lines)

        System.out.println(args[0]);

        File dir = new File(args[0]);
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File child : directoryListing) {
                System.out.println("Processing: " + child.getName());
                parseFile(child.getAbsolutePath(), child.getName(), map);
            }
        } else {
            System.out.println("Unable to scan directory: " + args[0]);
        }


        System.out.println("Start writing files down.");

        try {
            File outputDir = new File(args[1]);
            outputDir.mkdirs();

            System.out.println("Created path: " + outputDir.getAbsolutePath());

            for (String key : map.keySet()) {
                File file = new File(outputDir.getAbsolutePath() + File.separator + key + ".txt");
                file.createNewFile();

                BufferedWriter writer = new BufferedWriter(new FileWriter(file));

                for (List<String> line : map.get(key)){
                    writer.write(String.join(" ", line));
                    writer.newLine();
                }

                writer.flush();
                writer.close();

                System.out.println("Written file: " + key + ".txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void parseFile(String absolutePath, String name, Map<String, List<List<String>>> map) {
        String method = name.split("_")[0];

        String dataset;
        String seed;
        if ("bandit".equals(method)) {
            dataset = name.split("_")[2];
            seed = (name.split("_")[3]).split("\\.")[0];
        } else {
            dataset = name.split("_")[1];
            seed = (name.split("_")[2]).split("\\.")[0];
        }

        System.out.println("Method: " + method);
        System.out.println("Dataset: " + dataset);
        System.out.println("Seed: " + seed);


        map.putIfAbsent(dataset, new ArrayList<>()); // a map entry for each dataset
        List<String> oneLine = new ArrayList<>();
        oneLine.add(method);
        oneLine.add(seed);

        // add achieved metric's results to line:

        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(absolutePath));

            String line = "";
            while ((line = bufferedReader.readLine()) != null) {
                if (line.startsWith("Metric:")) {
                    oneLine.add(line.split(": ")[2]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        map.get(dataset).add(oneLine);

        if (oneLine.size() != 22) {
            System.err.println("Ready metrics: " + (oneLine.size()-2) + " Incomplete file: " + name);
        }

        System.out.println("Constructed a line for " + dataset + " of length " + oneLine.size());
        System.out.println();
    }
}
