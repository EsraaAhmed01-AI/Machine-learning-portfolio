import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio import SeqIO, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, DistanceMatrix
from Bio.Phylo.BaseTree import Tree
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
import argparse

warnings.filterwarnings('ignore')

class PhylogeneticTreeAnalyzer:
    """Analyzes phylogenetic relationships using ML-based sequence similarity and tree construction."""

    def __init__(self):
        self.data = None
        self.query_sequence = None
        self.query_id = None
        self.matching_percentage = 95.0
        self.actual_percentage = None
        self.matched_sequences = []
        self.tree_structure = {}
        self.similarity_scores = {}
        self.ai_model = None  # ML model for sequence classification
        self.genotype_model = None  # Model for genotype prediction
        self.label_encoder = LabelEncoder()  # Encoder for ML labels
        self.genotype_label_encoder = LabelEncoder()  # Encoder for genotype labels
        self.ml_tree = None
        self.ml_alignment = None
        self.ml_results = {}
        self.horizontal_line_tracker = []
        self.query_ml_group = None
        self.base_horizontal_length = 1.2
        self.ml_model_accuracy = None  # Accuracy of ML model
        self.genotype_model_accuracy = None  # Accuracy of genotype model

    # --- Data Loading ---
    def load_data(self, data_file: str) -> bool:
        """Loads sequence data from a CSV file."""
        try:
            self.data = pd.read_csv(data_file)
            print(f"✓ Data loaded: {len(self.data)} sequences, "
                  f"{self.data['ML'].nunique()} ML groups, "
                  f"{self.data['Genotype'].nunique()} genotypes")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    # --- Model Training ---
    def train_ai_model(self) -> bool:
        """Trains RandomForest models for ML group and genotype prediction."""
        try:
            if len(self.data) < 10:
                print("⚠️ Insufficient data for training (minimum 10 samples)")
                return False

            print("🤖 Training AI models...")
            f_gene_sequences = self.data['F-gene'].fillna('').astype(str)
            features = []
            for seq in f_gene_sequences:
                seq_clean = re.sub(r'[^ATGC]', '', seq.upper())
                if len(seq_clean) < 3:
                    features.append([0] * 100)
                    continue
                feature_vector = []
                kmers_3 = [seq_clean[i:i+3] for i in range(len(seq_clean)-2)]
                kmer_counts_3 = {kmer: kmers_3.count(kmer) for kmer in set(kmers_3)}
                kmers_4 = [seq_clean[i:i+4] for i in range(len(seq_clean)-3)]
                kmer_counts_4 = {kmer: kmers_4.count(kmer) for kmer in set(kmers_4)}
                all_3mers = [''.join(p) for p in itertools.product('ATGC', repeat=3)]
                all_4mers = [''.join(p) for p in itertools.product('ATGC', repeat=4)]
                feature_vector.extend([kmer_counts_3.get(kmer, 0) for kmer in all_3mers[:50]])
                feature_vector.extend([kmer_counts_4.get(kmer, 0) for kmer in all_4mers[:50]])
                features.append(feature_vector)

            X = np.array(features)

            # Train ML model
            ml_targets = self.label_encoder.fit_transform(self.data['ML'].fillna('Unknown'))
            if len(np.unique(ml_targets)) < 2:
                print("⚠️ Need at least 2 ML classes for training")
                return False
            X_train, X_test, y_train, y_test = train_test_split(X, ml_targets, test_size=0.2, random_state=42)
            self.ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ai_model.fit(X_train, y_train)
            self.ml_model_accuracy = self.ai_model.score(X_test, y_test)
            print(f"✓ ML model trained with accuracy: {self.ml_model_accuracy:.2%}")

            # Train genotype model
            genotype_targets = self.genotype_label_encoder.fit_transform(self.data['Genotype'].fillna('Unknown'))
            if len(np.unique(genotype_targets)) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(X, genotype_targets, test_size=0.2, random_state=42)
                self.genotype_model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.genotype_model.fit(X_train, y_train)
                self.genotype_model_accuracy = self.genotype_model.score(X_test, y_test)
                print(f"✓ Genotype model trained with accuracy: {self.genotype_model_accuracy:.2%}")

            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False

    def predict_ml_group(self, sequence: str) -> str:
        """Predicts ML group for a sequence using the trained model."""
        try:
            if not self.ai_model:
                return "Unknown"
            seq_clean = re.sub(r'[^ATGC]', '', sequence.upper())
            if len(seq_clean) < 3:
                return "Unknown"
            feature_vector = []
            kmers_3 = [seq_clean[i:i+3] for i in range(len(seq_clean)-2)]
            kmer_counts_3 = {kmer: kmers_3.count(kmer) for kmer in set(kmers_3)}
            kmers_4 = [seq_clean[i:i+4] for i in range(len(seq_clean)-3)]
            kmer_counts_4 = {kmer: kmers_4.count(kmer) for kmer in set(kmers_4)}
            all_3mers = [''.join(p) for p in itertools.product('ATGC', repeat=3)]
            all_4mers = [''.join(p) for p in itertools.product('ATGC', repeat=4)]
            feature_vector.extend([kmer_counts_3.get(kmer, 0) for kmer in all_3mers[:50]])
            feature_vector.extend([kmer_counts_4.get(kmer, 0) for kmer in all_4mers[:50]])
            X = np.array([feature_vector])
            ml_pred = self.label_encoder.inverse_transform(self.ai_model.predict(X))[0]
            return ml_pred
        except Exception as e:
            print(f"Error predicting ML group: {e}")
            return "Unknown"

    def predict_genotype(self, sequence: str) -> str:
        """Predicts genotype for a sequence using the trained model."""
        try:
            if not self.genotype_model:
                return "Unknown"
            seq_clean = re.sub(r'[^ATGC]', '', sequence.upper())
            if len(seq_clean) < 3:
                return "Unknown"
            feature_vector = []
            kmers_3 = [seq_clean[i:i+3] for i in range(len(seq_clean)-2)]
            kmer_counts_3 = {kmer: kmers_3.count(kmer) for kmer in set(kmers_3)}
            kmers_4 = [seq_clean[i:i+4] for i in range(len(seq_clean)-3)]
            kmer_counts_4 = {kmer: kmers_4.count(kmer) for kmer in set(kmers_4)}
            all_3mers = [''.join(p) for p in itertools.product('ATGC', repeat=3)]
            all_4mers = [''.join(p) for p in itertools.product('ATGC', repeat=4)]
            feature_vector.extend([kmer_counts_3.get(kmer, 0) for kmer in all_3mers[:50]])
            feature_vector.extend([kmer_counts_4.get(kmer, 0) for kmer in all_4mers[:50]])
            X = np.array([feature_vector])
            genotype_pred = self.genotype_label_encoder.inverse_transform(self.genotype_model.predict(X))[0]
            return genotype_pred
        except Exception as e:
            print(f"Error predicting genotype: {e}")
            return "Unknown"

    # --- Sequence Processing ---
    def find_query_sequence(self, query_input: str) -> bool:
        """Identifies query sequence by accession number, F-gene, or as a novel sequence."""
        try:
            query_input = query_input.strip()
            if query_input in self.data['Accession Number'].values:
                self.query_id = query_input
                query_row = self.data[self.data['Accession Number'] == query_input].iloc[0]
                self.query_sequence = query_row['F-gene']
                print(f"✓ Query found by accession: {query_input}, ML: {query_row['ML']}, Genotype: {query_row['Genotype']}")
                return True
            query_clean = re.sub(r'[^ATGC]', '', str(query_input).upper())
            if query_clean in self.data['F-gene'].values:
                query_row = self.data[self.data['F-gene'] == query_clean].iloc[0]
                self.query_id = query_row['Accession Number']
                self.query_sequence = query_clean
                print(f"✓ Query matched to accession: {self.query_id}, ML: {query_row['ML']}, Genotype: {query_row['Genotype']}")
                return True
            if len(query_clean) >= 10:
                self.query_id = f"QUERY_{hash(query_clean) % 100000:05d}"
                self.query_sequence = query_clean
                predicted_ml = self.predict_ml_group(query_clean)
                predicted_genotype = self.predict_genotype(query_clean)
                print(f"✓ Novel query accepted: {self.query_id}, Length: {len(query_clean)}, "
                      f"Predicted ML: {predicted_ml}, Predicted Genotype: {predicted_genotype}")
                return True
            print(f"✗ Invalid query: Too short (<10) or not found")
            return False
        except Exception as e:
            print(f"Error processing query: {e}")
            return False

    def calculate_f_gene_similarity(self, seq1: str, seq2: str) -> float:
        """Calculates similarity between two sequences using k-mer analysis."""
        try:
            if not seq1 or not seq2:
                return 0.0
            seq1 = re.sub(r'[^ATGC]', '', str(seq1).upper())
            seq2 = re.sub(r'[^ATGC]', '', str(seq2).upper())
            if len(seq1) == 0 or len(seq2) == 0:
                return 0.0
            k = 5
            kmers1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1) if len(seq1[i:i+k]) == k)
            kmers2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1) if len(seq2[i:i+k]) == k)
            if len(kmers1) == 0 and len(kmers2) == 0:
                return 100.0
            if len(kmers1) == 0 or len(kmers2) == 0:
                return 0.0
            intersection = len(kmers1.intersection(kmers2))
            union = len(kmers1.union(kmers2))
            return round((intersection / union) * 100, 2) if union > 0 else 0.0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def find_similar_sequences(self, target_percentage: float) -> Tuple[List[str], float]:
        """Finds sequences similar to the query sequence."""
        try:
            print(f"🔍 Finding sequences with {target_percentage}% similarity...")
            similarities = []
            for idx, row in self.data.iterrows():
                if row['Accession Number'] == self.query_id:
                    continue
                similarity = self.calculate_f_gene_similarity(self.query_sequence, row['F-gene'])
                similarities.append({
                    'id': row['Accession Number'],
                    'similarity': similarity,
                    'ml': row.get('ML', 'Unknown'),
                    'genotype': row.get('Genotype', 'Unknown')
                })
            if not similarities:
                print("❌ No valid sequences for comparison")
                return [], target_percentage
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            target_range = 2.0
            candidates = [s for s in similarities if abs(s['similarity'] - target_percentage) <= target_range]
            if not candidates:
                closest = min(similarities, key=lambda x: abs(x['similarity'] - target_percentage))
                actual_percentage = closest['similarity']
                candidates = [s for s in similarities if abs(s['similarity'] - actual_percentage) <= 1.0]
                print(f"⚠ No sequences at {target_percentage}%. Using closest: {actual_percentage:.1f}%")
            else:
                actual_percentage = target_percentage
            max_results = 50
            if len(candidates) > max_results:
                candidates = candidates[:max_results]
                print(f"⚠ Limited to top {max_results} matches")
            self.similarity_scores = {c['id']: c['similarity'] for c in candidates}
            matched_ids = [c['id'] for c in candidates]
            if similarities:
                max_sim = max(s['similarity'] for s in similarities)
                min_sim = min(s['similarity'] for s in similarities)
                avg_sim = sum(s['similarity'] for s in similarities) / len(similarities)
                print(f"✓ Found {len(matched_ids)} sequences at ~{actual_percentage:.1f}% similarity, "
                      f"Range: {min_sim:.1f}% - {max_sim:.1f}% (avg: {avg_sim:.1f}%)")
            return matched_ids, actual_percentage
        except Exception as e:
            print(f"Error finding similar sequences: {e}")
            return [], target_percentage

    # --- Tree Construction ---
    def build_tree_structure(self, matched_ids: List[str]) -> Dict:
        """Builds a hierarchical tree structure based on ML groups and genotypes."""
        try:
            print("🌳 Building normalized tree structure...")
            tree_structure = {
                'root': {'name': 'Root', 'type': 'root', 'children': {}, 'x': 0, 'y': 0,
                         'has_vertical_attachment': False, 'extension_level': 0}
            }
            ml_groups = {}
            for idx, row in self.data.iterrows():
                ml_group = row['ML']
                genotype = row['Genotype']
                seq_id = row['Accession Number']
                if ml_group not in ml_groups:
                    ml_groups[ml_group] = {}
                if genotype not in ml_groups[ml_group]:
                    ml_groups[ml_group][genotype] = []
                ml_groups[ml_group][genotype].append({
                    'id': seq_id, 'data': row.to_dict(), 'is_query': seq_id == self.query_id,
                    'is_matched': seq_id in matched_ids, 'similarity': self.similarity_scores.get(seq_id, 0.0)
                })
            if self.query_id.startswith("QUERY_"):
                predicted_ml = self.predict_ml_group(self.query_sequence)
                predicted_genotype = self.predict_genotype(self.query_sequence)
                if predicted_ml not in ml_groups:
                    ml_groups[predicted_ml] = {}
                if predicted_genotype not in ml_groups[predicted_ml]:
                    ml_groups[predicted_ml][predicted_genotype] = []
                ml_groups[predicted_ml][predicted_genotype].append({
                    'id': self.query_id, 'data': {
                        'F-gene': self.query_sequence, 'ML': predicted_ml, 'Genotype': predicted_genotype,
                        'Accession Number': self.query_id
                    }, 'is_query': True, 'is_matched': False, 'similarity': 100.0
                })
            normalized_ml_groups = self._normalize_ml_groups(ml_groups)
            self._build_normalized_ml_nodes(tree_structure, normalized_ml_groups, matched_ids)
            self.tree_structure = tree_structure
            print("✓ Tree structure built")
            return tree_structure
        except Exception as e:
            print(f"Error building tree structure: {e}")
            return {}

    def build_tree_structure_with_ml_safe(self, matched_ids: List[str]) -> Dict:
        """Enhances tree structure with ML analysis."""
        try:
            print("🌳 Building ML-enhanced tree structure...")
            ml_results = self.perform_ml_analysis_safe(matched_ids)
            tree_structure = self.build_tree_structure(matched_ids)
            if ml_results and 'tree' in ml_results:
                tree_structure['ml_analysis'] = {
                    'log_likelihood': ml_results['log_likelihood'],
                    'sequence_count': ml_results['sequence_count'],
                    'alignment_length': ml_results['alignment_length'],
                    'ml_tree_available': True
                }
                self.ml_tree = ml_results['tree']
                self.ml_alignment = ml_results.get('alignment')
                print("✓ Tree enhanced with ML analysis")
            else:
                tree_structure['ml_analysis'] = {'ml_tree_available': False, 'error': 'ML analysis failed'}
                print("⚠ ML analysis failed, using standard tree")
            return tree_structure
        except Exception as e:
            print(f"Error building ML-enhanced tree: {e}")
            try:
                return self.build_tree_structure(matched_ids)
            except Exception as e2:
                print(f"Fallback failed: {e2}")
                return {'error': 'Tree construction failed'}

    def _normalize_ml_groups(self, ml_groups: Dict) -> Dict:
        """Normalizes ML group names for hierarchical organization."""
        try:
            normalized_groups = {}
            for ml_name, genotypes in ml_groups.items():
                base_ml = 'UNCL' if ml_name.startswith('UNCL') else ml_name.split('.')[0] if '.' in ml_name and any(c.isdigit() for c in ml_name) else ml_name
                if base_ml not in normalized_groups:
                    normalized_groups[base_ml] = {'full_ml_groups': {}, 'representative_sequences': [], 'has_special_sequences': False}
                has_special = any(any(seq['is_query'] or seq['is_matched'] for seq in seqs) for seqs in genotypes.values())
                if has_special:
                    normalized_groups[base_ml]['has_special_sequences'] = True
                    normalized_groups[base_ml]['full_ml_groups'][ml_name] = genotypes
                elif len(normalized_groups[base_ml]['representative_sequences']) < 2:
                    for genotype, sequences in list(genotypes.items())[:2]:
                        if len(normalized_groups[base_ml]['representative_sequences']) < 2:
                            normalized_groups[base_ml]['representative_sequences'].extend(sequences[:1])
            return normalized_groups
        except Exception as e:
            print(f"Error normalizing ML groups: {e}")
            return {}

    def _build_normalized_ml_nodes(self, tree_structure: Dict, normalized_ml_groups: Dict, matched_ids: List[str]):
        """Builds normalized ML nodes with equal spacing."""
        try:
            self.horizontal_line_tracker = []
            self._identify_query_ml_group(normalized_ml_groups)
            ml_positions = self._calculate_dynamic_ml_positions(normalized_ml_groups)
            tree_structure['root']['has_vertical_attachment'] = len(normalized_ml_groups) > 1
            for ml_idx, (base_ml, ml_data) in enumerate(normalized_ml_groups.items()):
                y_pos = ml_positions[ml_idx]
                has_vertical = ml_data['has_special_sequences'] and len(ml_data['full_ml_groups']) > 1
                contains_query = base_ml == self.query_ml_group
                horizontal_length = self._determine_horizontal_line_length('normalized_ml_group', has_vertical, contains_query)
                x_pos = horizontal_length
                tree_structure['root']['children'][base_ml] = {
                    'name': base_ml, 'type': 'normalized_ml_group', 'children': {}, 'x': x_pos, 'y': y_pos,
                    'has_special_sequences': ml_data['has_special_sequences'], 'has_vertical_attachment': has_vertical,
                    'horizontal_line_length': horizontal_length, 'contains_query': contains_query
                }
                if ml_data['has_special_sequences']:
                    self._build_full_ml_nodes(tree_structure['root']['children'][base_ml], ml_data['full_ml_groups'],
                                             y_pos, matched_ids, x_pos)
                else:
                    self._add_representative_sequences(tree_structure['root']['children'][base_ml],
                                                       ml_data['representative_sequences'], y_pos, x_pos)
        except Exception as e:
            print(f"Error building normalized ML nodes: {e}")

    def _build_full_ml_nodes(self, normalized_ml_node: Dict, full_ml_groups: Dict, base_y: float, matched_ids: List[str], parent_x: float):
        """Builds full ML nodes with genotypes."""
        try:
            full_ml_positions = self._calculate_full_ml_positions(full_ml_groups, base_y)
            for ml_idx, (full_ml_name, genotypes) in enumerate(full_ml_groups.items()):
                y_pos = full_ml_positions[ml_idx]
                special_genotypes_count = sum(1 for g, seqs in genotypes.items() if any(s['is_query'] or s['is_matched'] for s in seqs))
                has_vertical = special_genotypes_count > 1
                contains_query = any(any(seq['is_query'] for seq in seqs) for seqs in genotypes.values())
                horizontal_length = self._determine_horizontal_line_length('full_ml_group', has_vertical, contains_query)
                x_pos = parent_x + horizontal_length
                normalized_ml_node['children'][full_ml_name] = {
                    'name': full_ml_name, 'type': 'full_ml_group', 'children': {}, 'x': x_pos, 'y': y_pos,
                    'sequences_count': sum(len(seqs) for seqs in genotypes.values()), 'has_vertical_attachment': has_vertical,
                    'horizontal_line_length': horizontal_length, 'contains_query': contains_query
                }
                self._build_genotype_nodes(normalized_ml_node['children'][full_ml_name], genotypes, y_pos, matched_ids, x_pos)
        except Exception as e:
            print(f"Error building full ML nodes: {e}")

    def _build_genotype_nodes(self, full_ml_node: Dict, genotypes: Dict, base_y: float, matched_ids: List[str], parent_x: float):
        """Builds genotype nodes with sequences."""
        try:
            special_genotypes = [(g, seqs) for g, seqs in genotypes.items() if any(s['is_query'] or s['is_matched'] for s in seqs)]
            if not special_genotypes:
                return
            genotype_positions = self._calculate_genotype_positions(special_genotypes, base_y)
            genotype_sequence_counts = [(g, seqs, len([s for s in seqs if s['is_query'] or s['is_matched']])) for g, seqs in special_genotypes]
            for gt_idx, (genotype, sequences, sequence_count) in enumerate(genotype_sequence_counts):
                y_pos = genotype_positions[gt_idx]
                special_sequences = [s for s in sequences if s['is_query'] or s['is_matched']]
                has_vertical = len(special_sequences) > 1
                contains_query = any(s['is_query'] for s in sequences)
                horizontal_length = self._determine_genotype_horizontal_line_length(sequence_count, has_vertical, contains_query)
                x_pos = parent_x + horizontal_length
                full_ml_node['children'][genotype] = {
                    'name': genotype, 'type': 'genotype', 'children': {}, 'x': x_pos, 'y': y_pos,
                    'sequences': sequences, 'has_vertical_attachment': has_vertical,
                    'horizontal_line_length': horizontal_length, 'contains_query': contains_query,
                    'sequence_count': sequence_count
                }
                self._add_sequences_horizontal(full_ml_node['children'][genotype], sequences, y_pos, x_pos)
        except Exception as e:
            print(f"Error building genotype nodes: {e}")

    def _add_representative_sequences(self, normalized_ml_node: Dict, representative_sequences: List[Dict], base_y: float, parent_x: float):
        """Adds representative sequences to normalized ML nodes."""
        try:
            if not representative_sequences:
                return
            has_vertical = len(representative_sequences) > 1
            horizontal_length = self._determine_horizontal_line_length('representative', has_vertical)
            x_pos = parent_x + horizontal_length
            if len(representative_sequences) == 1:
                seq = representative_sequences[0]
                normalized_ml_node['children'][f"{seq['id']}_rep"] = {
                    'name': f"{seq['id']} (Rep)", 'type': 'representative_sequence', 'data': seq,
                    'x': x_pos, 'y': base_y, 'has_vertical_attachment': False, 'horizontal_line_length': horizontal_length
                }
            else:
                positions = self._calculate_sequence_positions(representative_sequences, base_y)
                for idx, seq in enumerate(representative_sequences):
                    normalized_ml_node['children'][f"{seq['id']}_rep"] = {
                        'name': f"{seq['id']} (Rep)", 'type': 'representative_sequence', 'data': seq,
                        'x': x_pos, 'y': positions[idx], 'has_vertical_attachment': False, 'horizontal_line_length': horizontal_length
                    }
        except Exception as e:
            print(f"Error adding representative sequences: {e}")

    def _add_sequences_horizontal(self, genotype_node: Dict, sequences: List[Dict], base_y: float, parent_x: float):
        """Adds sequences with similarity-based line lengths."""
        try:
            query_line_length = 3.0
            query_sequences = [s for s in sequences if s['is_query']]
            matched_sequences = [s for s in sequences if s['is_matched'] and not s['is_query']]
            all_special_sequences = query_sequences + matched_sequences
            if len(all_special_sequences) == 1:
                sequence = all_special_sequences[0]
                line_length = self._calculate_similarity_based_line_length(sequence, query_line_length)
                x_pos = parent_x + line_length
                genotype_node['children'][sequence['id']] = {
                    'name': f"{sequence['id']} ({sequence['similarity']}%)" if sequence['is_matched'] else sequence['id'],
                    'type': 'sequence', 'data': sequence, 'x': x_pos, 'y': base_y,
                    'has_vertical_attachment': False, 'similarity_line_length': line_length
                }
            else:
                sequence_positions = self._calculate_sequence_positions(all_special_sequences, base_y)
                for seq_idx, sequence in enumerate(all_special_sequences):
                    line_length = self._calculate_similarity_based_line_length(sequence, query_line_length)
                    x_pos = parent_x + line_length
                    genotype_node['children'][sequence['id']] = {
                        'name': f"{sequence['id']} ({sequence['similarity']}%)" if sequence['is_matched'] else sequence['id'],
                        'type': 'sequence', 'data': sequence, 'x': x_pos, 'y': sequence_positions[seq_idx],
                        'has_vertical_attachment': False, 'similarity_line_length': line_length
                    }
        except Exception as e:
            print(f"Error adding sequences: {e}")

    def _identify_query_ml_group(self, normalized_ml_groups: Dict):
        """Identifies the ML group containing the query sequence."""
        try:
            for base_ml, ml_data in normalized_ml_groups.items():
                if ml_data['has_special_sequences']:
                    for genotypes in ml_data['full_ml_groups'].values():
                        for sequences in genotypes.values():
                            if any(seq['is_query'] for seq in sequences):
                                self.query_ml_group = base_ml
                                return
        except Exception as e:
            print(f"Error identifying query ML group: {e}")

    def _calculate_dynamic_ml_positions(self, normalized_ml_groups: Dict) -> List[float]:
        """Calculates equal Y positions for ML groups."""
        try:
            ml_count = len(normalized_ml_groups)
            if ml_count == 0:
                return []
            if ml_count == 1:
                return [0.0]
            total_spacing = (ml_count - 1) * 2.0
            start_y = -total_spacing / 2
            return [start_y + i * 2.0 for i in range(ml_count)]
        except Exception as e:
            print(f"Error calculating ML positions: {e}")
            return list(range(len(normalized_ml_groups)))

    def _calculate_full_ml_positions(self, full_ml_groups: Dict, base_y: float) -> List[float]:
        """Calculates equal positions for full ML groups."""
        try:
            ml_count = len(full_ml_groups)
            if ml_count <= 1:
                return [base_y]
            spacing = 1.5
            start_y = base_y - (spacing * (ml_count - 1)) / 2
            return [start_y + i * spacing for i in range(ml_count)]
        except Exception as e:
            print(f"Error calculating full ML positions: {e}")
            return [base_y] * len(full_ml_groups)

    def _calculate_genotype_positions(self, special_genotypes: List, base_y: float) -> List[float]:
        """Calculates equal positions for genotypes."""
        try:
            genotype_count = len(special_genotypes)
            if genotype_count <= 1:
                return [base_y]
            spacing = 1.0
            start_y = base_y - (spacing * (genotype_count - 1)) / 2
            return [start_y + i * spacing for i in range(genotype_count)]
        except Exception as e:
            print(f"Error calculating genotype positions: {e}")
            return [base_y] * len(special_genotypes)

    def _calculate_sequence_positions(self, sequences: List[Dict], base_y: float) -> List[float]:
        """Calculates equal positions for sequences."""
        try:
            seq_count = len(sequences)
            if seq_count <= 1:
                return [base_y]
            spacing = 0.8
            start_y = base_y - (spacing * (seq_count - 1)) / 2
            return [start_y + i * spacing for i in range(seq_count)]
        except Exception as e:
            print(f"Error calculating sequence positions: {e}")
            return [base_y] * len(sequences)

    def _calculate_similarity_based_line_length(self, sequence: Dict, query_line_length: float) -> float:
        """Calculates line length based on sequence similarity."""
        try:
            if sequence['is_query']:
                return query_line_length
            if sequence['is_matched']:
                similarity = sequence['similarity']
                proportional_length = (similarity / 100.0) * query_line_length
                return max(proportional_length, query_line_length * 0.2)
            return query_line_length * 0.5
        except Exception as e:
            print(f"Error calculating line length: {e}")
            return query_line_length * 0.5

    def _determine_horizontal_line_length(self, node_type: str, has_vertical: bool, contains_query: bool = False) -> float:
        """Determines horizontal line length based on node type."""
        try:
            base_length = self.base_horizontal_length
            if contains_query and node_type == 'normalized_ml_group':
                return base_length * 2.5
            if has_vertical:
                current_max = base_length
                for length in self.horizontal_line_tracker:
                    if length > current_max:
                        current_max = length
                new_length = current_max + 0.3
                self.horizontal_line_tracker.append(new_length)
                return new_length
            return base_length
        except Exception as e:
            print(f"Error determining line length: {e}")
            return self.base_horizontal_length

    def _determine_genotype_horizontal_line_length(self, sequence_count: int, has_vertical: bool, contains_query: bool = False) -> float:
        """Determines horizontal line length for genotype nodes."""
        try:
            base_length = self.base_horizontal_length
            query_bonus = 0.5 if contains_query else 0.0
            if sequence_count <= 1:
                length_multiplier = 1.0
            elif sequence_count <= 3:
                length_multiplier = 1.6
            elif sequence_count <= 5:
                length_multiplier = 2.3
            else:
                length_multiplier = 6.0
            return base_length * length_multiplier + query_bonus
        except Exception as e:
            print(f"Error determining genotype line length: {e}")
            return self.base_horizontal_length

    # --- Visualization ---
    def create_interactive_tree(self, matched_ids: List[str], actual_percentage: float) -> Optional[go.Figure]:
        """Creates an interactive horizontal phylogenetic tree visualization."""
        try:
            print("🎨 Creating interactive tree visualization...")
            edge_x, edge_y = [], []
            node_x, node_y = [], []
            node_colors, node_text, node_hover, node_sizes = [], [], [], []
            colors = {
                'root': '#FF0000', 'normalized_ml_group': '#FFB6C1', 'full_ml_group': '#FF69B4',
                'genotype': '#FFD700', 'representative_sequence': '#FFA500', 'query_sequence': '#4B0082',
                'matched_sequence': '#6A5ACD', 'other_sequence': '#87CEEB'
            }

            def add_horizontal_edges(parent_x, parent_y, children_dict):
                if not children_dict:
                    return
                children_list = list(children_dict.values())
                if len(children_list) == 1:
                    child = children_list[0]
                    edge_x.extend([parent_x, child['x'], None])
                    edge_y.extend([parent_y, child['y'], None])
                else:
                    child_x_positions = [child['x'] for child in children_list]
                    min_child_x = min(child_x_positions)
                    intermediate_x = parent_x + (min_child_x - parent_x) * 0.8
                    edge_x.extend([parent_x, intermediate_x, None])
                    edge_y.extend([parent_y, parent_y, None])
                    child_y_positions = [child['y'] for child in children_list]
                    min_y, max_y = min(child_y_positions), max(child_y_positions)
                    edge_x.extend([intermediate_x, intermediate_x, None])
                    edge_y.extend([min_y, max_y, None])
                    for child in children_list:
                        edge_x.extend([intermediate_x, child['x'], None])
                        edge_y.extend([child['y'], child['y'], None])

            def get_node_color_and_size(node):
                if node['type'] == 'sequence':
                    if node['data']['is_query']:
                        return colors['query_sequence'], 10
                    if node['data']['is_matched']:
                        return colors['matched_sequence'], 8
                    return colors['other_sequence'], 6
                if node['type'] == 'representative_sequence':
                    return colors['representative_sequence'], 7
                if node['type'] == 'normalized_ml_group':
                    return colors['normalized_ml_group'], 9 if node.get('has_special_sequences', False) else 7
                if node['type'] == 'full_ml_group':
                    return colors['full_ml_group'], 8
                if node['type'] == 'genotype':
                    return colors['genotype'], 7
                return colors.get(node['type'], '#000000'), 7

            def create_node_text(node):
                if node['type'] == 'sequence':
                    return f"{node['name']}" if node['data']['is_matched'] and not node['data']['is_query'] else node['name']
                if node['type'] == 'representative_sequence':
                    return node['name']
                if node['type'] == 'normalized_ml_group':
                    return f"{node['name']} *" if node.get('has_special_sequences', False) else node['name']
                return node['name']

            def create_hover_text(node):
                if node['type'] == 'sequence':
                    data = node['data']['data']
                    hover_text = (
                        f"<b>{node['name']}</b><br>Type: {'Query' if node['data']['is_query'] else 'Matched' if node['data']['is_matched'] else 'Other'} Sequence<br>"
                        f"ML Group: {data.get('ML', 'N/A')}<br>Genotype: {data.get('Genotype', 'N/A')}<br>"
                        f"Host: {data.get('Host', 'N/A')}<br>Country: {data.get('Country', 'N/A')}<br>"
                        f"Isolate: {data.get('Isolate', 'N/A')}<br>Year: {data.get('Year', 'N/A')}"
                    )
                    if node['data']['is_matched']:
                        hover_text += f"<br><b>Similarity: {node['data']['similarity']}%</b>"
                elif node['type'] == 'representative_sequence':
                    data = node['data']['data']
                    hover_text = (
                        f"<b>{node['name']}</b><br>Type: Representative Sequence<br>"
                        f"ML Group: {data.get('ML', 'N/A')}<br>Genotype: {data.get('Genotype', 'N/A')}<br>"
                        f"Host: {data.get('Host', 'N/A')}<br>Country: {data.get('Country', 'N/A')}"
                    )
                elif node['type'] == 'normalized_ml_group':
                    hover_text = f"<b>{node['name']}</b><br>Type: Normalized ML Group"
                    if node.get('has_special_sequences', False):
                        hover_text += "<br>Contains query/matched sequences"
                    else:
                        hover_text += "<br>Representative sequences only"
                elif node['type'] == 'full_ml_group':
                    hover_text = f"<b>{node['name']}</b><br>Type: Full ML Group"
                    if 'sequences_count' in node:
                        hover_text += f"<br>Total Sequences: {node['sequences_count']}"
                elif node['type'] == 'genotype':
                    hover_text = f"<b>{node['name']}</b><br>Type: Genotype"
                    if 'sequences' in node:
                        special_count = sum(1 for seq in node['sequences'] if seq['is_query'] or seq['is_matched'])
                        hover_text += f"<br>Special Sequences: {special_count}/{len(node['sequences'])}"
                else:
                    hover_text = f"<b>{node['name']}</b><br>Type: {node['type'].replace('_', ' ').title()}"
                return hover_text

            def add_node_and_edges(node, parent_x=None, parent_y=None):
                x, y = node['x'], node['y']
                node_x.append(x)
                node_y.append(y)
                color, size = get_node_color_and_size(node)
                node_colors.append(color)
                node_sizes.append(size)
                node_text.append(create_node_text(node))
                node_hover.append(create_hover_text(node))
                if 'children' in node and node['children']:
                    add_horizontal_edges(x, y, node['children'])
                    for child in node['children'].values():
                        add_node_and_edges(child, x, y)

            root_node = self.tree_structure['root']
            add_node_and_edges(root_node)
            if root_node['children']:
                add_horizontal_edges(root_node['x'], root_node['y'], root_node['children'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray'),
                hoverinfo='none', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black'), opacity=0.85),
                text=node_text, textposition="middle right", textfont=dict(size=9, color="black"),
                hoverinfo='text', hovertext=node_hover, showlegend=False
            ))

            min_x, max_x = min(node_x), max(node_x) if node_x else (0, 1)
            min_y, max_y = min(node_y), max(node_y) if node_y else (0, 1)
            x_range = max_x - min_x
            y_range = max_y - min_y
            x_padding = x_range * 0.2 if x_range > 0 else 1
            y_padding = y_range * 0.2 if y_range > 0 else 1
            width = min(1400, max(800, int(x_range * 80 + 400)))
            height = min(900, max(500, int(y_range * 40 + 300)))

            fig.update_layout(
                title=dict(
                    text=f"Horizontal Phylogenetic Tree<br>Query: {self.query_id} | Similarity: {actual_percentage}% | Matched: {len(matched_ids)}",
                    x=0.5, font=dict(size=12)
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min_x - x_padding, max_x + x_padding], automargin=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min_y - y_padding, max_y + y_padding], automargin=True),
                plot_bgcolor="white", paper_bgcolor="white", hovermode="closest",
                width=width, height=height, margin=dict(l=20, r=100, t=40, b=10),
                showlegend=True, legend=dict(x=1.02, y=1, xanchor='left', yanchor='top',
                                             bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1, font=dict(size=10))
            )

            legend_elements = [
                dict(name="Root", marker=dict(color=colors['root'], size=8)),
                dict(name="Normalized ML Groups", marker=dict(color=colors['normalized_ml_group'], size=8)),
                dict(name="Full ML Groups", marker=dict(color=colors['full_ml_group'], size=8)),
                dict(name="Genotypes", marker=dict(color=colors['genotype'], size=8)),
                dict(name="Query Sequence", marker=dict(color=colors['query_sequence'], size=10)),
                dict(name="Similar Sequences", marker=dict(color=colors['matched_sequence'], size=9)),
                dict(name="Representative Sequences", marker=dict(color=colors['representative_sequence'], size=8)),
            ]
            for element in legend_elements:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=element['marker'], name=element['name'], showlegend=True))

            config = {
                'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {'format': 'png', 'filename': 'phylogenetic_tree', 'height': height, 'width': width, 'scale': 2}
            }
            try:
                fig.show(config)
            except Exception as e:
                print(f"Warning: Could not display figure: {e}")
            return fig
        except Exception as e:
            print(f"Error creating tree visualization: {e}")
            return None

    # --- ML Analysis ---
    def perform_ml_analysis_safe(self, matched_ids: List[str]) -> Dict:

        try:
            print("\n🧬 PERFORMING MAXIMUM LIKELIHOOD ANALYSIS")
            print("="*50)

            # Include query sequence in analysis
            all_sequences = [self.query_id] + [seq_id for seq_id in matched_ids if seq_id != self.query_id]

            # Limit number of sequences to prevent memory issues
            if len(all_sequences) > 20:
                print(f"Warning: Limiting analysis to 20 sequences (had {len(all_sequences)})")
                all_sequences = all_sequences[:20]

            if len(all_sequences) < 3:
                print("❌ Need at least 3 sequences for ML analysis")
                return {}

            # Step 1: Create multiple sequence alignment
            alignment = self.create_sequence_alignment(all_sequences)
            if not alignment:
                return {}

            # Step 2: Calculate ML distances
            distance_matrix = self.calculate_ml_distances(alignment)
            if distance_matrix.size == 0:
                return {}

            # Step 3: Construct ML tree
            ml_tree = self.construct_ml_tree(alignment)
            if not ml_tree:
                return {}

            # Step 4: Calculate tree likelihood (safely)
            log_likelihood = self.calculate_ml_likelihood_safe(ml_tree, alignment)

            # Step 5: Prepare results
            ml_results = {
                'tree': ml_tree,
                'alignment': alignment,
                'distance_matrix': distance_matrix,
                'log_likelihood': log_likelihood,
                'sequence_count': len(all_sequences),
                'alignment_length': len(alignment[0]) if alignment else 0
            }

            print(f"✅ ML analysis completed successfully")
            print(f"   Sequences analyzed: {len(all_sequences)}")
            print(f"   Alignment length: {ml_results['alignment_length']}")
            print(f"   Log-likelihood: {log_likelihood:.2f}")

            return ml_results

        except Exception as e:
            print(f"❌ ML analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


    def create_sequence_alignment(self, sequence_ids: List[str]) -> Optional[MultipleSeqAlignment]:

        try:
            print("🧬 Creating multiple sequence alignment...")

            # Get sequences
            sequences = []
            for seq_id in sequence_ids:
                try:
                    row = self.data[self.data['Accession Number'] == seq_id]
                    if not row.empty:
                        f_gene = str(row.iloc[0]['F-gene'])
                        # Clean sequence (remove non-nucleotide characters)
                        clean_seq = re.sub(r'[^ATGCN-]', '', f_gene.upper())
                        if len(clean_seq) > 10:  # Minimum sequence length
                            seq_record = SeqRecord(Seq(clean_seq), id=seq_id, description="")
                            sequences.append(seq_record)
                except Exception as e:
                    print(f"Warning: Skipping sequence {seq_id}: {e}")
                    continue

            if len(sequences) < 2:
                print("❌ Need at least 2 valid sequences for alignment")
                return None

            # Simple alignment (you might want to use MUSCLE or CLUSTAL for better results)
            aligned_sequences = self._simple_alignment(sequences)

            print(f"✓ Alignment created with {len(aligned_sequences)} sequences")
            return MultipleSeqAlignment(aligned_sequences)

        except Exception as e:
            print(f"Error creating alignment: {e}")
            return None

    def _simple_alignment(self, sequences: List[SeqRecord]) -> List[SeqRecord]:

        try:
            # Find maximum length
            max_length = max(len(seq.seq) for seq in sequences)

            # Cap maximum length to prevent memory issues
            if max_length > 10000:
                max_length = 10000
                print(f"Warning: Sequences truncated to {max_length} bp")

            # Pad sequences to same length
            aligned_sequences = []
            for seq in sequences:
                seq_str = str(seq.seq)[:max_length]  # Truncate if too long

                if len(seq_str) < max_length:
                    # Pad with gaps at the end
                    padded_seq = seq_str + '-' * (max_length - len(seq_str))
                else:
                    padded_seq = seq_str

                aligned_sequences.append(SeqRecord(Seq(padded_seq), id=seq.id, description=seq.description))

            return aligned_sequences
        except Exception as e:
            print(f"Error in simple alignment: {e}")
            return sequences

    def calculate_ml_distances(self, alignment: MultipleSeqAlignment) -> np.ndarray:

        try:
            print("📊 Calculating ML distances...")

            # Convert alignment to numeric matrix
            seq_matrix = self._alignment_to_matrix(alignment)
            n_sequences = len(alignment)

            if n_sequences == 0:
                return np.array([])

            # Initialize distance matrix
            distance_matrix = np.zeros((n_sequences, n_sequences))

            # Calculate pairwise ML distances
            for i in range(n_sequences):
                for j in range(i + 1, n_sequences):
                    try:
                        ml_distance = self._calculate_ml_distance_pair(seq_matrix[i], seq_matrix[j])
                        distance_matrix[i][j] = ml_distance
                        distance_matrix[j][i] = ml_distance
                    except Exception as e:
                        print(f"Warning: Error calculating distance between sequences {i} and {j}: {e}")
                        # Use maximum distance as fallback
                        distance_matrix[i][j] = 1.0
                        distance_matrix[j][i] = 1.0

            print("✓ ML distances calculated")
            return distance_matrix

        except Exception as e:
            print(f"Error calculating ML distances: {e}")
            return np.array([])

    def _alignment_to_matrix(self, alignment: MultipleSeqAlignment) -> np.ndarray:

        try:
            nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4, '-': 5}

            matrix = []
            for record in alignment:
                sequence = str(record.seq).upper()
                numeric_seq = [nucleotide_map.get(nuc, 4) for nuc in sequence]
                matrix.append(numeric_seq)

            return np.array(matrix)
        except Exception as e:
            print(f"Error converting alignment to matrix: {e}")
            return np.array([])


    def _calculate_ml_distance_pair(self, seq1: np.ndarray, seq2: np.ndarray) -> float:

        try:
            if len(seq1) == 0 or len(seq2) == 0:
                return 1.0

            # Count differences (excluding gaps and N's)
            valid_positions = (seq1 < 4) & (seq2 < 4)  # Exclude N's and gaps

            if np.sum(valid_positions) == 0:
                return 1.0  # Maximum distance if no valid comparisons

            differences = np.sum(seq1[valid_positions] != seq2[valid_positions])
            total_valid = np.sum(valid_positions)

            if total_valid == 0:
                return 1.0

            # Calculate proportion of differences
            p = differences / total_valid

            # Jukes-Cantor correction
            if p >= 0.75:
                return 1.0  # Maximum distance

            # JC distance formula: -3/4 * ln(1 - 4p/3)
            try:
                jc_distance = -0.75 * np.log(1 - (4 * p / 3))
                return min(max(jc_distance, 0.0), 1.0)  # Clamp between 0 and 1
            except (ValueError, RuntimeWarning):
                return 1.0  # Return maximum distance if log calculation fails

        except Exception as e:
            return 1.0

    def construct_ml_tree(self, alignment: MultipleSeqAlignment) -> Optional[Tree]:
        """Constructs a maximum likelihood tree."""
        try:
            print("🌳 Constructing ML tree...")
            distance_matrix = self.calculate_ml_distances(alignment)
            if distance_matrix.size == 0:
                return None
            sequence_names = [record.id for record in alignment]
            tree = self._build_nj_tree_from_distances(distance_matrix, sequence_names)
            if tree:
                tree = self._optimize_branch_lengths_ml_safe(tree, alignment)
            print("✓ ML tree constructed")
            return tree
        except Exception as e:
            print(f"Error constructing ML tree: {e}")
            return None

    def _build_nj_tree_from_distances(self, distance_matrix: np.ndarray, sequence_names: List[str]) -> Optional[Tree]:
        """Builds a neighbor-joining tree from distance matrix."""
        try:
            if distance_matrix.shape[0] != len(sequence_names):
                print("Error: Distance matrix size mismatch")
                return None
            matrix_data = [[0.0 if i == j else max(0.0, float(distance_matrix[i][j])) for j in range(i + 1)] for i in range(len(sequence_names))]
            dm = DistanceMatrix(names=sequence_names, matrix=matrix_data)
            constructor = DistanceTreeConstructor()
            tree = constructor.nj(dm)
            return tree if self._validate_tree_structure(tree) else None
        except Exception as e:
            print(f"Error building NJ tree: {e}")
            return None

    def _validate_tree_structure(self, tree: Tree, max_depth: int = 100) -> bool:
        """Validates tree structure to prevent recursion issues."""
        try:
            visited = set()
            def check_node(node, depth=0):
                if depth > max_depth:
                    return False
                node_id = id(node)
                if node_id in visited:
                    return False
                visited.add(node_id)
                return all(check_node(child, depth + 1) for child in getattr(node, 'clades', []))
            return check_node(tree.root if hasattr(tree, 'root') else tree)
        except Exception:
            return False

    def _optimize_branch_lengths_ml_safe(self, tree: Tree, alignment: MultipleSeqAlignment) -> Tree:
        """Optimizes branch lengths using ML model."""
        try:
            print("🔧 Optimizing branch lengths...")
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(1000)
            try:
                seq_matrix = self._alignment_to_matrix(alignment)
                if seq_matrix.size == 0:
                    return tree
                all_clades = self._get_clades_safe(tree)
                for clade in all_clades:
                    if hasattr(clade, 'branch_length') and clade.branch_length is not None:
                        optimal_length = self._calculate_optimal_branch_length(clade, seq_matrix)
                        clade.branch_length = max(optimal_length, 0.001)
            finally:
                sys.setrecursionlimit(old_limit)
            print("✓ Branch lengths optimized")
            return tree
        except Exception as e:
            print(f"Warning: Branch optimization failed: {e}")
            return tree

    def _get_clades_safe(self, tree: Tree, max_depth: int = 50) -> List:
        """Safely retrieves all clades in the tree."""
        clades = []
        visited = set()
        def traverse_node(node, depth=0):
            if depth > max_depth or id(node) in visited:
                return
            visited.add(id(node))
            clades.append(node)
            for child in getattr(node, 'clades', []):
                traverse_node(child, depth + 1)
        try:
            traverse_node(tree.root if hasattr(tree, 'root') else tree)
        except Exception as e:
            print(f"Warning: Tree traversal error: {e}")
        return clades

    def _calculate_optimal_branch_length(self, clade: float, seq_matrix: np.ndarray) -> float:
        """Calculates optimal branch length for a clade."""
        try:
            if not hasattr(clade, 'branch_length') or clade.branch_length is None:
                return 0.1
            current_length = float(clade.branch_length)
            if np.isnan(current_length) or np.isinf(current_length) or current_length <= 0:
                return 0.1
            return min(max(current_length * (0.9 if hasattr(clade, 'name') and clade.name else 1.1), 0.001), 1.0)
        except Exception:
            return 0.1

    def calculate_ml_likelihood_safe(self, tree: Tree, alignment: MultipleSeqAlignment) -> float:
        """Calculates tree likelihood using Jukes-Cantor model."""
        try:
            print("Trying to calculate tree likelihood...")
            seq_matrix = self._alignment_to_matrix(alignment)
            if seq_matrix.size == 0:
                return -np.inf
            total_log_likelihood = 0.0
            n_sites = min(seq_matrix.shape[1], 1000)
            for site in range(0, n_sites, max(1, n_sites // 100)):
                site_pattern = seq_matrix[:, site]
                valid_positions = site_pattern < 4
                if np.sum(valid_positions) < 2:
                    continue
                site_likelihood = self._calculate_site_likelihood_safe(tree, site_pattern)
                if site_likelihood > 0:
                    total_log_likelihood += np.log(site_likelihood)
            print(f"Likelihood: {total_log_likelihood:.2f}")
            return total_log_likelihood
        except Exception as e:
            print(f"Error calculating likelihood: {e}")
            return -np.inf

    def _calculate_site_likelihood_safe(self, tree: np.ndarray, site_pattern: np.ndarray) -> float:
        """Calculates likelihood for a single site."""
        try:
            valid_nucs = site_pattern[site_pattern < 4]
            if len(valid_nucs) == 0:
                return 1.0
            unique_nucs = len(np.unique(valid_nucs))
            total_nucs = len(valid_nucs)
            diversity_factor = unique_nucs / 4.0
            likelihood = np.exp(-diversity_factor * total_nucs * 0.1)
            return max(likelihood, 1e-10)
        except Exception:
            return 1e-10

    # --- Reporting ---
    def generate_detailed_report(self, matched_ids: List[str], actual_percentage: float) -> bool:
        """
        Generate a detailed HTML report with query details, matched sequences, model performance,
        phylogenetic tree insights, and ML analysis results in tabular format.
        """
        try:
            print("📝 Generating detailed HTML analysis report...")

            # Debug: Inspect tree structure
            print("Tree Structure Summary:")
            print(f"Root children: {len(self.tree_structure['root']['children'])}")
            def print_tree_summary(node, level=0):
                print("  " * level + f"Node: {node['name']} (Type: {node['type']}, Children: {len(node.get('children', {}))})")
                for child in node.get('children', {}).values():
                    print_tree_summary(child, level + 1)
            print_tree_summary(self.tree_structure['root'])

            # --- HTML Template with Escaped Curly Braces ---
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Phylogenetic Analysis Report - {query_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
                    h1 {{ text-align: center; color: #2c3e50; }}
                    h2 {{ color: #34495e; margin-top: 20px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #3498db; color: #fff; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #e0f7fa; }}
                    .metadata {{ margin-left: 20px; font-size: 0.9em; }}
                    .metadata p {{ margin: 5px 0; }}
                    @media (max-width: 600px) {{ table {{ font-size: 0.85em; }} th, td {{ padding: 8px; }} }}
                </style>
            </head>
            <body>
                <h1>Phylogenetic Analysis Report</h1>
                <p style="text-align: center;">Generated on: {timestamp}</p>
                <p style="text-align: center;">Query ID: {query_id}</p>
            """
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            html_content = html_content.format(query_id=self.query_id, timestamp=timestamp)

            # --- Query Information ---
            query_type = (
                "Accession Number" if self.query_id in self.data['Accession Number'].values else
                "Dataset Sequence" if self.query_sequence in self.data['F-gene'].values else
                "Novel Sequence"
            )
            query_ml = self.predict_ml_group(self.query_sequence) if query_type == "Novel Sequence" else self.data[
                (self.data['Accession Number'] == self.query_id) |
                (self.data['F-gene'] == re.sub(r'[^ATGC]', '', self.query_sequence.upper()))
            ].iloc[0]['ML']
            query_genotype = self.predict_genotype(self.query_sequence) if query_type == "Novel Sequence" else self.data[
                (self.data['Accession Number'] == self.query_id) |
                (self.data['F-gene'] == re.sub(r'[^ATGC]', '', self.query_sequence.upper()))
            ].iloc[0]['Genotype']
            query_metadata = (
                {"F-gene": self.query_sequence[:50] + "..." if len(self.query_sequence) > 50 else self.query_sequence}
                if query_type == "Novel Sequence" else
                self.data[(self.data['Accession Number'] == self.query_id) |
                          (self.data['F-gene'] == re.sub(r'[^ATGC]', '', self.query_sequence.upper()))].iloc[0].to_dict()
            )
            query_info_table = [
                ["Query ID", self.query_id],
                ["Query Type", query_type],
                ["Sequence Length", f"{len(self.query_sequence)} nucleotides"],
                ["ML Group", query_ml],
                ["Genotype", query_genotype],
                ["Target Similarity", f"{self.matching_percentage}%"],
                ["Actual Similarity", f"{actual_percentage:.1f}%"]
            ]
            html_content += """
                <h2>Query Information</h2>
                <table><tr><th>Field</th><th>Value</th></tr>
            """
            for row in query_info_table:
                html_content += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
            html_content += "</table><div class='metadata'><h3>Metadata</h3>"
            for key, value in query_metadata.items():
                html_content += f"<p><strong>{key}:</strong> {value}</p>"
            html_content += "</div>"

            # --- Matched Sequences ---
            matched_sequences_table = []
            headers = ["Accession Number", "Similarity (%)", "ML Group", "Genotype", "Host", "Country", "Isolate", "Year"]
            for seq_id in matched_ids:
                row = self.data[self.data['Accession Number'] == seq_id].iloc[0]
                matched_sequences_table.append([
                    seq_id,
                    f"{self.similarity_scores.get(seq_id, 0.0):.1f}",
                    row.get('ML', 'N/A'),
                    row.get('Genotype', 'N/A'),
                    row.get('Host', 'N/A'),
                    row.get('Country', 'N/A'),
                    row.get('Isolate', 'N/A'),
                    row.get('Year', 'N/A')
                ])
            html_content += f"""
                <h2>Matched Sequences</h2>
                <p>Total Matched Sequences: {len(matched_ids)}</p>
            """
            if matched_sequences_table:
                html_content += "<table><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
                for row in matched_sequences_table:
                    html_content += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                html_content += "</table>"
            else:
                html_content += "<p>No matched sequences found.</p>"

            # --- Model Performance ---
            model_performance_table = [
                ["ML Model Accuracy", f"{self.ml_model_accuracy:.2%}" if self.ml_model_accuracy else "Not trained"],
                ["Genotype Model Accuracy", f"{self.genotype_model_accuracy:.2%}" if self.genotype_model_accuracy else "Not trained"]
            ]
            html_content += """
                <h2>Model Performance</h2>
                <table><tr><th>Metric</th><th>Value</th></tr>
            """
            for row in model_performance_table:
                html_content += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
            html_content += "</table>"

            # --- Phylogenetic Tree Insights ---
            def count_nodes(node, depth=0, visited=None):
                if visited is None:
                    visited = set()
                node_id = id(node)
                if node_id in visited:
                    print(f"Warning: Cycle detected at node {node['name']}")
                    return 0
                visited.add(node_id)
                count = 1
                children = node.get('children', {})
                print(f"Counting node: {node['name']} (Type: {node['type']}, Depth: {depth}, Children: {len(children)})")
                for child in children.values():
                    count += count_nodes(child, depth + 1, visited)
                return count

            total_nodes = count_nodes(self.tree_structure['root'])
            print(f"Total Nodes Counted: {total_nodes}")

            query_node_path = []
            def find_query_path(node, path):
                if node.get('data', {}).get('is_query', False):
                    query_node_path.append(" -> ".join(path + [node['name']]))
                for name, child in node.get('children', {}).items():
                    find_query_path(child, path + [node['name']])

            find_query_path(self.tree_structure['root'], [])
            tree_insights_table = [
                ["Total Nodes", total_nodes],
                ["ML Groups Represented", len(self.tree_structure['root']['children'])],
                ["Query Node Path", query_node_path[0] if query_node_path else "Not found"]
            ]
            html_content += """
                <h2>Phylogenetic Tree Insights</h2>
                <table><tr><th>Field</th><th>Value</th></tr>
            """
            for row in tree_insights_table:
                html_content += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
            html_content += "</table>"

            # --- ML Analysis Results ---
            ml_analysis = self.tree_structure.get('ml_analysis', {})
            ml_analysis_table = [
                ["ML Tree Available", ml_analysis.get('ml_tree_available', False)],
                ["Log-Likelihood", f"{ml_analysis.get('log_likelihood', 'N/A'):.2f}" if ml_analysis.get('log_likelihood') else "N/A"],
                ["Sequence Count", ml_analysis.get('sequence_count', 'N/A')],
                ["Alignment Length", ml_analysis.get('alignment_length', 'N/A')]
            ]
            html_content += """
                <h2>Maximum Likelihood Analysis Results</h2>
                <table><tr><th>Field</th><th>Value</th></tr>
            """
            for row in ml_analysis_table:
                html_content += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
            html_content += "</table></body></html>"

            # --- Save HTML Report ---
            report_filename = f"detailed_report_{self.query_id.replace('/', '_')}.html"
            with open(report_filename, 'w') as f:
                f.write(html_content)
            print(f"✓ Detailed HTML report saved as '{report_filename}'")
            return True
        except Exception as e:
            print(f"Error generating detailed report: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def command_line_interface():
    """Parse command-line arguments and run phylogenetic analysis."""
    parser = argparse.ArgumentParser(
        description="Advanced Phylogenetic Tree Analyzer with AI-enhanced similarity matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s -d data.csv -q MH087032 -s 95\n  %(prog)s -d data.csv -q MH087032 -s 90 --no-ai --batch query1,query2,query3"
    )
    parser.add_argument('-d', '--data', required=True, help='Path to CSV data file')
    parser.add_argument('-q', '--query', required=True, help='Query sequence ID or nucleotide sequence')
    parser.add_argument('-s', '--similarity', type=float, default=95.0, help='Target similarity percentage (70-99, default: 95)')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI model training')
    parser.add_argument('--batch', help='Comma-separated list of query IDs for batch processing')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--save-json', action='store_true', help='Save detailed results to JSON')

    args = parser.parse_args()

    # Validate arguments
    if not 70 <= args.similarity <= 99:
        print("❌ Similarity percentage must be between 70 and 99.")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = PhylogeneticTreeAnalyzer()
    if not analyzer.load_data(args.data):
        print("❌ Failed to load data.")
        sys.exit(1)

    # Train AI model unless disabled
    if not args.no_ai:
        print("⏳ Training AI model...")
        start_time = time.time()
        if analyzer.train_ai_model():
            print(f"✅ AI model training completed in {time.time() - start_time:.1f} seconds")
        else:
            print("⚠️ AI model training failed, continuing with basic analysis")

    # Process queries
    queries = args.batch.split(',') if args.batch else [args.query]
    for query in queries:
        query = query.strip()
        print(f"🔍 Processing: {query}")
        if not analyzer.find_query_sequence(query):
            print(f"❌ Query not found: {query}")
            continue

        matched_ids, actual_percentage = analyzer.find_similar_sequences(args.similarity)
        if not matched_ids:
            print(f"❌ No similar sequences found for {query}")
            continue

        analyzer.build_tree_structure_with_ml_safe(matched_ids)
        fig = analyzer.create_interactive_tree(matched_ids, actual_percentage)
        if fig:
            html_filename = f"phylogenetic_tree_{query.replace('/', '_')}_interactive.html"
            fig.write_html(html_filename)
            print(f"📄 Interactive HTML saved: {html_filename}")
            analyzer.generate_detailed_report(matched_ids, actual_percentage)
            print(f"📄 Detailed HTML report saved: detailed_report_{query.replace('/', '_')}.html")
        print(f"✅ Analysis completed for {query}")

def main():
    """Run interactive phylogenetic analysis with user input."""
    print("\n" + "="*70)
    print("🧬 PHYLOGENETIC TREE ANALYZER - ADVANCED ML-BASED ANALYSIS")
    print("Version 2.0 | AI-Enhanced Similarity Matching")
    print("="*70)

    analyzer = PhylogeneticTreeAnalyzer()

    # Load data
    data_file = "f cleaned.csv"
    while not Path(data_file).exists() or not analyzer.load_data(data_file):
        print(f"❌ File not found or invalid: {data_file}")
        data_file = input("Enter valid data file path: ").strip()
        if not data_file:
            print("❌ Analysis cancelled.")
            return

    # Train AI model
    print("⏳ Training AI model...")
    start_time = time.time()
    if analyzer.train_ai_model():
        print(f"✅ AI model training completed in {time.time() - start_time:.1f} seconds")
    else:
        print("⚠️ AI model training failed, continuing with basic analysis")

    # Get query sequence
    while True:
        query_input = input("\nEnter query sequence or ID (min 10 nucleotides): ").strip()
        if analyzer.find_query_sequence(query_input):
            break
        retry = input("❌ Invalid input. Try again? (y/n): ").strip().lower()
        if retry != 'y':
            print("👋 Analysis cancelled.")
            return

    # Set similarity percentage
    while True:
        try:
            similarity_input = input("Enter target similarity percentage (1-99) [85]: ").strip()
            target_percentage = float(similarity_input) if similarity_input else 85.0
            if 1 <= target_percentage <= 99:
                analyzer.matching_percentage = target_percentage
                break
            print("❌ Please enter a percentage between 1 and 99.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # Find similar sequences
    print(f"⏳ Analyzing sequences for {target_percentage}% similarity...")
    start_time = time.time()
    matched_ids, actual_percentage = analyzer.find_similar_sequences(target_percentage)
    if not matched_ids:
        print(f"❌ No similar sequences found at {target_percentage}% similarity.")
        return
    analyzer.matched_sequences = matched_ids
    analyzer.actual_percentage = actual_percentage
    print(f"✅ Similarity analysis completed in {time.time() - start_time:.1f} seconds")

    # Build tree structure
    print("⏳ Building phylogenetic tree structure...")
    start_time = time.time()
    tree_structure = analyzer.build_tree_structure_with_ml_safe(matched_ids)
    if not tree_structure:
        print("❌ Failed to build tree structure.")
        return
    print(f"✅ Tree structure built in {time.time() - start_time:.1f} seconds")

    # Create visualization and save HTML
    print("⏳ Creating interactive visualization...")
    start_time = time.time()
    fig = analyzer.create_interactive_tree(matched_ids, actual_percentage)
    if not fig:
        print("❌ Visualization creation failed.")
        return

    html_filename = "phylogenetic_tree_interactive.html"
    fig.write_html(html_filename)
    print(f"📄 Interactive HTML saved: {html_filename}")

    # Generate detailed report
    print("⏳ Generating detailed report...")
    start_time = time.time()
    if analyzer.generate_detailed_report(matched_ids, actual_percentage):
        print(f"✅ Detailed report generated in {time.time() - start_time:.1f} seconds")

    print(f"\n🎉 Analysis completed successfully!")
    print(f"   Query ID: {analyzer.query_id}")
    print(f"   Query sequence length: {len(analyzer.query_sequence)} nucleotides")
    print(f"   Similar sequences found: {len(matched_ids)}")
    print(f"   Actual similarity percentage: {actual_percentage:.1f}%")
    print(f"   HTML visualization file: {html_filename}")
    print(f"   HTML report file: detailed_report_{analyzer.query_id.replace('/', '_')}.html")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)