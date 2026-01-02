// Hybrid One-Parameter-at-a-Time Strategy (HOS)
import java.io.*;
import java.util.*;
import java.util.StringTokenizer;

public class HOS {

    // GA parameters
    static final int GA_POPULATION_SIZE = 100;
    static final int GA_MAX_GENERATIONS = 1000;
    static final double GA_CROSSOVER_RATE = 0.5;
    static final double GA_MUTATION_RATE = 0.03;

    // SA parameters
    static final double SA_T0 = 0.1;
    static final double SA_ALPHA = 0.85;
    static final int SA_MAX_ITER = 5000;

    public static void main(String[] args) throws Exception {
        ArrayList<String> Configuration = new ArrayList<>();
        Configuration.addAll(Arrays.asList("Test.txt"));
         for(int n = 0; n < Configuration.size(); n++){
                    String inputPath = "C:\\Users\\mbuma\\Desktop\\PWGASA\\HOA\\" + Configuration.get(n);
        String outputDir = "C:\\Users\\mbuma\\Desktop\\PWGASA\\HOA\\";
        String resultsCsv = outputDir + "Results_Summary.csv";

        // Read parameters and build ParIntegerVal and LongestPair once
        File file = new File(inputPath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);

        int interaction_Strength = Integer.parseInt(br.readLine().trim());

        ArrayList<String> parameters = new ArrayList<>();
        ArrayList<String> values = new ArrayList<>();
        ArrayList<Integer> length = new ArrayList<>();
        ArrayList<String> LongestPair = new ArrayList<>();
        ArrayList<ArrayList<Integer>> ParIntegerVal = new ArrayList<>();

        String line;
        StringTokenizer tokens;
        int j = 0;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            tokens = new StringTokenizer(line, ":");
            if (tokens.countTokens() < 2) continue;
            length.add(tokens.countTokens() - 1);
            parameters.add(tokens.nextToken().trim());
            while (tokens.hasMoreTokens()) {
                values.add(j, tokens.nextToken().trim());
                j++;
            }
        }
        // Map values to integer indices
        int initial = 0;
        for (int x = 0; x < length.size(); x++) {
            ArrayList<Integer> list = new ArrayList<>();
            int counter = length.get(x);
            for (int y = 0; y < counter; y++) {
                list.add(initial);
                initial++;
            }
            ParIntegerVal.add(list);
        }

        // Sort parameters by descending size for longest pair selection
        for (int r = 0; r < ParIntegerVal.size(); r++) {
            for (int s = r + 1; s < ParIntegerVal.size(); s++) {
                if (ParIntegerVal.get(r).size() < ParIntegerVal.get(s).size()) {
                    ArrayList<Integer> tmp = ParIntegerVal.get(r);
                    ParIntegerVal.set(r, ParIntegerVal.get(s));
                    ParIntegerVal.set(s, tmp);
                }
            }
        }

        if (interaction_Strength == 2) {
            for (int x = 0; x < ParIntegerVal.get(0).size(); x++) {
                for (int y = 0; y < ParIntegerVal.get(1).size(); y++) {
                    LongestPair.add(ParIntegerVal.get(0).get(x) + ":" + ParIntegerVal.get(1).get(y));
                }
            }
        }

        br.close();
        fr.close();

        System.out.println("Parameters integer values: " + ParIntegerVal);
        System.out.println("LongestPair size (initial): " + LongestPair.size());

        // Run 20 independent trials and record sizes
        int runs = 20;
        ArrayList<Integer> sizes = new ArrayList<>();
        for (int run = 1; run <= runs; run++) {
            System.out.println("\n========== RUN " + run + " ==========");
            ArrayList<String> finalSuite = runSingleTrial(LongestPair, ParIntegerVal, interaction_Strength);
            sizes.add(finalSuite.size());
            // Save per-run final suite file
            String perRunPath = outputDir + "FinalTestSuite_Run" + run + ".txt";
            saveFinalSuiteToFile(finalSuite, perRunPath);
            System.out.println("Run " + run + " final suite size: " + finalSuite.size());
        }

        int best = Collections.min(sizes);
        double avg = sizes.stream().mapToInt(Integer::intValue).average().orElse(0.0);

        System.out.println("\n========== SUMMARY ==========");
        System.out.println("Best test suite size over " + runs + " runs: " + best);
        System.out.println("Average test suite size over " + runs + " runs: " + String.format("%.2f", avg));

        // Append summary to CSV
        try (FileWriter fw = new FileWriter(resultsCsv, true)) {
            fw.write(new File(inputPath).getName() + "," + best + "," + String.format("%.2f", avg) + "\n");
            System.out.println("Results appended to: " + resultsCsv);
        } catch (IOException e) {
            System.err.println("Failed to write results CSV: " + e.getMessage());
        }
         }
    }

    // Run one full GA+SA+deduplication+check trial and return final deduplicated suite
    private static ArrayList<String> runSingleTrial(ArrayList<String> LongestPair,
            ArrayList<ArrayList<Integer>> ParIntegerVal, int interaction_Strength) {

        // 1) Horizontal Growth (GA)
        ArrayList<String> gaResult = performHorizontalGrowthGA(LongestPair, ParIntegerVal, interaction_Strength);

        // 2) Check missing pairs and run SA if needed
        int missingAfterGA = countMissingPairs(gaResult, ParIntegerVal);
        System.out.println("Missing pairs after GA: " + missingAfterGA);
        ArrayList<String> saResult = new ArrayList<>(gaResult);
        if (interaction_Strength == 2 && missingAfterGA > 0) {
            saResult = performVerticalGrowthSA(gaResult, ParIntegerVal, SA_T0, SA_ALPHA, SA_MAX_ITER);
            int missingAfterSA = countMissingPairs(saResult, ParIntegerVal);
            System.out.println("Missing pairs after SA: " + missingAfterSA);
        }

        // 3) Merge & Deduplicate
        ArrayList<String> merged = mergeAndDeduplicate(gaResult, saResult);
        int missingAfterMerge = countMissingPairs(merged, ParIntegerVal);
        System.out.println("Missing pairs after merge & dedup: " + missingAfterMerge);

        // 4) Refinement loop: re-run GA+SA up to max attempts if still missing
        int attempts = 0;
        int maxAttempts = 3;
        while (missingAfterMerge > 0 && attempts < maxAttempts) {
            attempts++;
            System.out.println("Refinement attempt " + attempts + " via GA+SA...");
            ArrayList<String> gaRefined = performHorizontalGrowthGA(LongestPair, ParIntegerVal, interaction_Strength);
            ArrayList<String> saRefined = gaRefined;
            if (countMissingPairs(gaRefined, ParIntegerVal) > 0) {
                saRefined = performVerticalGrowthSA(gaRefined, ParIntegerVal, SA_T0, SA_ALPHA, SA_MAX_ITER / 2);
            }
            merged = mergeAndDeduplicate(gaRefined, saRefined);
            missingAfterMerge = countMissingPairs(merged, ParIntegerVal);
            System.out.println("Missing after refinement: " + missingAfterMerge);
            if (missingAfterMerge == 0) break;
        }

        return merged;
    }

    // ---------------- GA implementation ----------------
    private static ArrayList<String> performHorizontalGrowthGA(ArrayList<String> LongestPair,
            ArrayList<ArrayList<Integer>> ParIntegerVal, int interactionStrength) {

        int populationSize = GA_POPULATION_SIZE;
        int maxGenerations = GA_MAX_GENERATIONS;
        double crossoverRate = GA_CROSSOVER_RATE;
        double mutationRate = GA_MUTATION_RATE;

        ArrayList<ArrayList<String>> population = initializePopulation(LongestPair, ParIntegerVal, populationSize);

        for (int generation = 0; generation < maxGenerations; generation++) {
            ArrayList<ArrayList<String>> newPopulation = new ArrayList<>();
            while (newPopulation.size() < populationSize) {
                ArrayList<String> parent1 = tournamentSelection(population, LongestPair);
                ArrayList<String> parent2 = tournamentSelection(population, LongestPair);
                ArrayList<String> child;
                if (Math.random() < crossoverRate) {
                    child = crossover(parent1, parent2);
                } else {
                    child = new ArrayList<>(parent1);
                }
                if (Math.random() < mutationRate) {
                    mutate(child, ParIntegerVal);
                }
                newPopulation.add(child);
            }
            population = newPopulation;
        }

        ArrayList<String> best = getBestSolution(population, LongestPair);
        // ensure non-null
        if (best == null) best = new ArrayList<>(LongestPair);
        return best;
    }

    private static ArrayList<ArrayList<String>> initializePopulation(ArrayList<String> LongestPair,
            ArrayList<ArrayList<Integer>> ParIntegerVal, int populationSize) {

        ArrayList<ArrayList<String>> population = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < populationSize; i++) {
            ArrayList<String> individual = new ArrayList<>(LongestPair);
            for (int j = 2; j < ParIntegerVal.size(); j++) {
                for (int k = 0; k < individual.size(); k++) {
                    if (j < ParIntegerVal.size() && ParIntegerVal.get(j).size() > 0) {
                        individual.set(k, individual.get(k) + ":" + ParIntegerVal.get(j).get(rand.nextInt(ParIntegerVal.get(j).size())));
                    }
                }
            }
            population.add(individual);
        }

        return population;
    }

    private static ArrayList<String> tournamentSelection(ArrayList<ArrayList<String>> population, ArrayList<String> LongestPair) {
        int tournamentSize = 2;
        ArrayList<ArrayList<String>> tournament = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < tournamentSize; i++) {
            tournament.add(population.get(random.nextInt(population.size())));
        }
        return getBestSolution(tournament, LongestPair);
    }

    private static ArrayList<String> crossover(ArrayList<String> parent1, ArrayList<String> parent2) {
        ArrayList<String> child = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < parent1.size(); i++) {
            if (random.nextBoolean()) child.add(parent1.get(i));
            else child.add(parent2.get(i));
        }
        return child;
    }

    private static void mutate(ArrayList<String> individual, ArrayList<ArrayList<Integer>> ParIntegerVal) {
        Random random = new Random();
        if (individual.size() == 0) return;
        int index = random.nextInt(individual.size());
        String raw = individual.get(index);
        String[] elements = raw.split(":");

        int paramCount = ParIntegerVal.size();

        if (elements.length > paramCount) {
            elements = Arrays.copyOf(elements, paramCount);
        } else if (elements.length < paramCount) {
            String[] newEl = new String[paramCount];
            System.arraycopy(elements, 0, newEl, 0, elements.length);
            for (int e = elements.length; e < paramCount; e++) {
                ArrayList<Integer> vals = ParIntegerVal.get(e);
                newEl[e] = String.valueOf(vals.get(new Random().nextInt(vals.size())));
            }
            elements = newEl;
        }

        int parameterIndex = new Random().nextInt(paramCount);
        ArrayList<Integer> valuesForParam = ParIntegerVal.get(parameterIndex);
        elements[parameterIndex] = String.valueOf(valuesForParam.get(new Random().nextInt(valuesForParam.size())));

        StringBuilder mutated = new StringBuilder();
        for (int i = 0; i < elements.length; i++) {
            mutated.append(elements[i]);
            if (i < elements.length - 1) mutated.append(":");
        }
        individual.set(index, mutated.toString());
    }

    private static ArrayList<String> getBestSolution(ArrayList<ArrayList<String>> population, ArrayList<String> LongestPair) {
        ArrayList<String> bestSolution = null;
        int bestFitness = Integer.MIN_VALUE;
        for (ArrayList<String> individual : population) {
            int fitness = evaluateFitness(individual, LongestPair);
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestSolution = individual;
            }
        }
        return bestSolution;
    }

    private static int evaluateFitness(ArrayList<String> individual, ArrayList<String> LongestPair) {
        Set<String> covered = new HashSet<>(LongestPair);
        covered.addAll(individual);
        return covered.size();
    }

    // ---------------- SA implementation (exploitation-only) ----------------
    private static int countMissingPairs(ArrayList<String> suite, ArrayList<ArrayList<Integer>> ParIntegerVal) {
        int paramCount = ParIntegerVal.size();
        int missing = 0;
        for (int i = 0; i < paramCount; i++) {
            for (int j = i + 1; j < paramCount; j++) {
                HashSet<String> allPairs = new HashSet<>();
                for (Integer a : ParIntegerVal.get(i)) {
                    for (Integer b : ParIntegerVal.get(j)) {
                        allPairs.add(a + ":" + b);
                    }
                }
                HashSet<String> covered = new HashSet<>();
                for (String test : suite) {
                    String[] elems = test.split(":");
                    if (elems.length > j) {
                        covered.add(elems[i] + ":" + elems[j]);
                    }
                }
                for (String p : allPairs) if (!covered.contains(p)) missing++;
            }
        }
        return missing;
    }

    private static ArrayList<String> performVerticalGrowthSA(ArrayList<String> initialSuite,
            ArrayList<ArrayList<Integer>> ParIntegerVal, double T0, double alpha, int maxIter) {

        Random rand = new Random();
        ArrayList<String> bestSuite = new ArrayList<>(initialSuite);
        int bestMissing = countMissingPairs(bestSuite, ParIntegerVal);
        double T = T0;
        int paramCount = ParIntegerVal.size();

        for (int iter = 0; iter < maxIter && T > 1e-6 && bestMissing > 0; iter++) {
            ArrayList<String> neighbor = new ArrayList<>(bestSuite);

            if (neighbor.size() == 0) break;
            int tcIndex = rand.nextInt(neighbor.size());
            String raw = neighbor.get(tcIndex);
            String[] elems = raw.split(":");
            if (elems.length != paramCount) {
                String[] newEl = new String[paramCount];
                System.arraycopy(elems, 0, newEl, 0, Math.min(elems.length, paramCount));
                for (int e = elems.length; e < paramCount; e++) {
                    ArrayList<Integer> vals = ParIntegerVal.get(e);
                    newEl[e] = String.valueOf(vals.get(rand.nextInt(vals.size())));
                }
                elems = newEl;
            }

            int pIndex = rand.nextInt(paramCount);
            ArrayList<Integer> valsForP = ParIntegerVal.get(pIndex);
            elems[pIndex] = String.valueOf(valsForP.get(rand.nextInt(valsForP.size())));

            StringBuilder sb = new StringBuilder();
            for (int z = 0; z < elems.length; z++) {
                sb.append(elems[z]);
                if (z < elems.length - 1) sb.append(":");
            }
            neighbor.set(tcIndex, sb.toString());

            int neighborMissing = countMissingPairs(neighbor, ParIntegerVal);
            if (neighborMissing < bestMissing) {
                bestSuite = neighbor;
                bestMissing = neighborMissing;
            } else {
                double prob = Math.exp((bestMissing - neighborMissing) / Math.max(1e-9, T));
                if (rand.nextDouble() < prob) {
                    bestSuite = neighbor;
                    bestMissing = neighborMissing;
                }
            }
            T *= alpha;
        }
        return bestSuite;
    }

    // ---------------- merging, deduplication, and I/O ----------------
    private static ArrayList<String> mergeAndDeduplicate(ArrayList<String> listA, ArrayList<String> listB) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        if (listA != null) set.addAll(listA);
        if (listB != null) set.addAll(listB);
        return new ArrayList<>(set);
    }

    private static void saveFinalSuiteToFile(ArrayList<String> suite, String outputPath) {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            for (String tc : suite) pw.println(tc);
            System.out.println("Final suite saved to: " + outputPath);
        } catch (IOException e) {
            System.err.println("Failed to save final suite: " + e.getMessage());
        }
    }

} // end class
