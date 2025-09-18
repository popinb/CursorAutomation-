# R Analysis Code for Qualtrics Latency Perception Study

# Load required packages
library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(emmeans)
library(broom.mixed)

# Load data (assuming CSV export from Qualtrics)
# data <- read.csv("qualtrics_export.csv", stringsAsFactors = FALSE)

# Data preprocessing function
prepare_data <- function(data) {
  # Convert to long format for trial data
  trial_data <- data %>%
    select(ResponseId, starts_with("trial")) %>%
    pivot_longer(
      cols = starts_with("trial"),
      names_to = c("trial_num", "variable"),
      names_pattern = "trial(\\d+)_(.*)",
      values_to = "value"
    ) %>%
    pivot_wider(
      names_from = variable,
      values_from = value
    ) %>%
    mutate(
      prompt_id = as.factor(prompt_id),
      modality = as.factor(modality),
      latency_sec = as.numeric(latency_ms) / 1000,
      latency_level = factor(case_when(
        latency_ms == 500 ~ "Fast",
        latency_ms == 2000 ~ "Medium",
        latency_ms == 6000 ~ "Slow"
      ), levels = c("Fast", "Medium", "Slow"))
    )
  
  # Extract matrix responses (Q2-Q6)
  matrix_data <- data %>%
    select(ResponseId, matches("Q[2-6]_\\d+")) %>%
    pivot_longer(
      cols = -ResponseId,
      names_to = c("question", "statement"),
      names_pattern = "(Q\\d+)_(\\d+)",
      values_to = "rating"
    ) %>%
    mutate(
      trial_num = as.numeric(substr(question, 2, 2)) - 1,
      statement = as.numeric(statement)
    )
  
  # Merge trial config with ratings
  analysis_data <- trial_data %>%
    left_join(matrix_data, by = c("ResponseId", "trial_num")) %>%
    pivot_wider(
      names_from = statement,
      names_prefix = "rating_",
      values_from = rating
    ) %>%
    rename(
      quality = rating_1,
      wait_felt_long = rating_2,
      wait_hurt_quality = rating_3,
      would_accept = rating_4
    ) %>%
    mutate(
      accept_binary = as.numeric(would_accept >= 5),
      participant_id = as.factor(ResponseId)
    )
  
  # Add perception slider data
  perception_data <- data %>%
    select(ResponseId, Q7_1:Q7_5) %>%
    pivot_longer(
      cols = Q7_1:Q7_5,
      names_to = "trial_letter",
      names_prefix = "Q7_",
      values_to = "perceived_sec"
    ) %>%
    mutate(
      trial_letter = LETTERS[as.numeric(trial_letter)]
    )
  
  # Parse trial letter mapping and merge
  # This would need the embedded data field trial_letter_mapping
  
  return(analysis_data)
}

# Analysis 1: Effect of latency and streaming on quality
analyze_quality <- function(data) {
  # Mixed effects model
  model_quality <- lmer(quality ~ latency_level * modality + 
                         (1 | participant_id) + (1 | prompt_id), 
                       data = data)
  
  # Summary
  print(summary(model_quality))
  
  # Estimated marginal means
  emm_quality <- emmeans(model_quality, ~ latency_level * modality)
  print(emm_quality)
  
  # Contrasts
  contrasts <- contrast(emm_quality, method = "pairwise")
  print(contrasts)
  
  # Plot
  plot_quality <- ggplot(data, aes(x = latency_level, y = quality, 
                                   color = modality, group = modality)) +
    stat_summary(fun = mean, geom = "point", size = 3) +
    stat_summary(fun = mean, geom = "line", size = 1) +
    stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.1) +
    scale_color_manual(values = c("nonstream" = "#E74C3C", "stream" = "#3498DB"),
                      labels = c("Non-streaming", "Streaming")) +
    labs(x = "Latency Level", y = "Quality Rating (1-7)", 
         color = "Response Mode",
         title = "Effect of Latency and Streaming on Perceived Quality") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(plot_quality)
  
  return(list(model = model_quality, plot = plot_quality))
}

# Analysis 2: Acceptance threshold (ℓ50)
analyze_acceptance <- function(data) {
  # Logistic mixed model
  model_accept <- glmer(accept_binary ~ latency_sec * modality + 
                         (1 | participant_id) + (1 | prompt_id), 
                       family = binomial, data = data)
  
  print(summary(model_accept))
  
  # Calculate ℓ50 for each modality
  # Create prediction grid
  pred_grid <- expand.grid(
    latency_sec = seq(0, 6, by = 0.1),
    modality = c("nonstream", "stream")
  )
  
  # Get predictions
  pred_grid$prob_accept <- predict(model_accept, 
                                   newdata = pred_grid, 
                                   type = "response",
                                   re.form = NA)
  
  # Find ℓ50 (latency where P(accept) = 0.5)
  l50_values <- pred_grid %>%
    group_by(modality) %>%
    summarize(
      l50 = approx(prob_accept, latency_sec, xout = 0.5)$y
    )
  
  print("ℓ50 values (seconds):")
  print(l50_values)
  print(paste("Streaming benefit:", 
              round(l50_values$l50[2] - l50_values$l50[1], 2), 
              "seconds"))
  
  # Plot acceptance curves
  plot_accept <- ggplot(pred_grid, aes(x = latency_sec, y = prob_accept, 
                                       color = modality)) +
    geom_line(size = 1.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5) +
    geom_vline(data = l50_values, aes(xintercept = l50, color = modality),
               linetype = "dotted", size = 1) +
    scale_color_manual(values = c("nonstream" = "#E74C3C", "stream" = "#3498DB"),
                      labels = c("Non-streaming", "Streaming")) +
    labs(x = "Latency (seconds)", y = "Probability of Acceptance",
         color = "Response Mode",
         title = "Acceptance Probability by Latency and Response Mode") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(plot_accept)
  
  return(list(model = model_accept, l50 = l50_values, plot = plot_accept))
}

# Analysis 3: Perceived vs actual latency
analyze_perception <- function(data) {
  # Calculate perception bias
  data_perception <- data %>%
    mutate(
      perception_bias = perceived_sec - latency_sec
    )
  
  # Mixed model for perception bias
  model_perception <- lmer(perception_bias ~ latency_level * modality + 
                            (1 | participant_id) + (1 | prompt_id), 
                          data = data_perception)
  
  print(summary(model_perception))
  
  # Plot
  plot_perception <- ggplot(data_perception, 
                           aes(x = latency_sec, y = perceived_sec, 
                               color = modality)) +
    geom_point(alpha = 0.3, position = position_jitter(width = 0.1)) +
    geom_smooth(method = "lm", se = TRUE) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", alpha = 0.5) +
    scale_color_manual(values = c("nonstream" = "#E74C3C", "stream" = "#3498DB"),
                      labels = c("Non-streaming", "Streaming")) +
    labs(x = "Actual Latency (seconds)", y = "Perceived Latency (seconds)",
         color = "Response Mode",
         title = "Perceived vs Actual Latency by Response Mode") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(plot_perception)
  
  # Summary stats
  perception_summary <- data_perception %>%
    group_by(modality, latency_level) %>%
    summarize(
      mean_bias = mean(perception_bias, na.rm = TRUE),
      se_bias = sd(perception_bias, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  print("Perception bias by condition:")
  print(perception_summary)
  
  return(list(model = model_perception, plot = plot_perception, 
              summary = perception_summary))
}

# Analysis 4: Overall preferences and thresholds
analyze_preferences <- function(data) {
  # Streaming preference (Q9)
  pref_summary <- data %>%
    distinct(ResponseId, .keep_all = TRUE) %>%
    count(Q9) %>%
    mutate(percentage = n / sum(n) * 100)
  
  print("Streaming preferences:")
  print(pref_summary)
  
  # Maximum acceptable wait (Q8)
  max_wait_summary <- data %>%
    distinct(ResponseId, .keep_all = TRUE) %>%
    summarize(
      mean_max_wait = mean(Q8, na.rm = TRUE),
      median_max_wait = median(Q8, na.rm = TRUE),
      sd_max_wait = sd(Q8, na.rm = TRUE),
      q25 = quantile(Q8, 0.25, na.rm = TRUE),
      q75 = quantile(Q8, 0.75, na.rm = TRUE)
    )
  
  print("Maximum acceptable wait time:")
  print(max_wait_summary)
  
  # Histogram of max acceptable wait
  plot_max_wait <- ggplot(data %>% distinct(ResponseId, .keep_all = TRUE), 
                          aes(x = Q8)) +
    geom_histogram(binwidth = 0.5, fill = "#3498DB", alpha = 0.7) +
    geom_vline(xintercept = max_wait_summary$median_max_wait, 
               color = "red", linetype = "dashed", size = 1) +
    labs(x = "Maximum Acceptable Wait (seconds)", y = "Count",
         title = "Distribution of Maximum Acceptable Wait Times",
         subtitle = paste("Median =", round(max_wait_summary$median_max_wait, 1), 
                         "seconds")) +
    theme_minimal()
  
  print(plot_max_wait)
  
  return(list(preferences = pref_summary, max_wait = max_wait_summary, 
              plot = plot_max_wait))
}

# Main analysis pipeline
run_analysis <- function(data) {
  # Prepare data
  analysis_data <- prepare_data(data)
  
  # Run all analyses
  results <- list(
    quality = analyze_quality(analysis_data),
    acceptance = analyze_acceptance(analysis_data),
    perception = analyze_perception(analysis_data),
    preferences = analyze_preferences(data)
  )
  
  # Create summary report
  cat("\n========== SUMMARY REPORT ==========\n")
  
  # Quality effects
  cat("\n1. Quality Ratings:\n")
  cat("   - Effect of latency on quality ratings\n")
  cat("   - Streaming benefit at each latency level\n")
  
  # Acceptance thresholds
  cat("\n2. Acceptance Thresholds (ℓ50):\n")
  cat(paste("   - Non-streaming:", round(results$acceptance$l50$l50[1], 2), "seconds\n"))
  cat(paste("   - Streaming:", round(results$acceptance$l50$l50[2], 2), "seconds\n"))
  cat(paste("   - Benefit:", round(results$acceptance$l50$l50[2] - 
                                   results$acceptance$l50$l50[1], 2), "seconds\n"))
  
  # Perception bias
  cat("\n3. Perception Bias:\n")
  cat("   - How streaming affects perceived wait times\n")
  
  # Preferences
  cat("\n4. User Preferences:\n")
  cat(paste("   - Median max acceptable wait:", 
            round(results$preferences$max_wait$median_max_wait, 1), "seconds\n"))
  
  return(results)
}

# Example usage:
# results <- run_analysis(data)

# Save all plots
save_plots <- function(results) {
  ggsave("quality_by_latency.png", results$quality$plot, 
         width = 8, height = 6, dpi = 300)
  ggsave("acceptance_curves.png", results$acceptance$plot, 
         width = 8, height = 6, dpi = 300)
  ggsave("perception_bias.png", results$perception$plot, 
         width = 8, height = 6, dpi = 300)
  ggsave("max_wait_distribution.png", results$preferences$plot, 
         width = 8, height = 6, dpi = 300)
}

# Power analysis function
power_analysis <- function() {
  library(simr)
  
  # Simulate data structure
  n_participants <- 100
  n_trials <- 5
  
  # Create simulated dataset
  sim_data <- expand.grid(
    participant_id = factor(1:n_participants),
    trial = 1:n_trials
  ) %>%
    mutate(
      prompt_id = factor(sample(1:5, n(), replace = TRUE)),
      modality = factor(sample(c("stream", "nonstream"), n(), replace = TRUE)),
      latency_level = factor(sample(c("Fast", "Medium", "Slow"), n(), replace = TRUE)),
      quality = 5 + rnorm(n())  # Baseline quality
    )
  
  # Fit model for power analysis
  model_sim <- lmer(quality ~ latency_level * modality + 
                     (1 | participant_id) + (1 | prompt_id), 
                   data = sim_data)
  
  # Test power for main effects
  power_latency <- powerSim(model_sim, 
                           test = fixed("latency_level"),
                           nsim = 100)
  
  power_modality <- powerSim(model_sim, 
                            test = fixed("modality"),
                            nsim = 100)
  
  print("Power Analysis Results:")
  print(power_latency)
  print(power_modality)
}